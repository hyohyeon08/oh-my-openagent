import { afterEach, describe, expect, test } from "bun:test"
import { EventEmitter, once } from "node:events"

import { buildCompletionDetails } from "../completion/notification"
import { InProcessRunner, type ChildSession, type ChildSessionEvent, type ChildSessionListener } from "../runners/in-process"
import { createTaskRecordStore, type TaskRecordStore } from "../store"
import { createTaskManager } from "./manager"
import { createInProcessManagedRunner } from "./runner"
import { FakeRunner, baseSpec, categoryPlanner, cleanupProjects, flush, makeManager, settings, tempProject } from "./__fixtures__/manager-fakes"

afterEach(cleanupProjects)

async function startedChild() {
  const processRunner = new FakeRunner()
  const { manager, store } = makeManager({ process: processRunner })
  const started = await manager.start(baseSpec({ execution_mode: "process" }))
  if (started.kind !== "started") throw new Error("expected started")
  const fake = processRunner.handles.get(started.task_id)
  if (fake === undefined) throw new Error("the runner produced no handle")
  return { manager, store, fake, taskId: started.task_id }
}

describe("a child that resumes on its own after its turn settled (omo#9069)", () => {
  test("#given a completed InProcessRunner child #when its session starts autonomous turns #then the manager reopens each epoch and records the outcome at agent_idle", async () => {
    // given the real runner and adapter, with only the child session controlled
    const project = tempProject()
    const persisted = createTaskRecordStore({ project_dir: project })
    const events = new EventEmitter()
    const store: TaskRecordStore = {
      ...persisted,
      appendEvent(taskId, event) {
        const result = persisted.appendEvent(taskId, event)
        events.emit(event.type)
        return result
      },
    }
    const listeners = new Set<ChildSessionListener>()
    const emit = (event: ChildSessionEvent): void => {
      for (const listener of listeners) listener(event)
    }
    let lastText: string | undefined
    let resolvePrompt: (() => void) | undefined
    const session: ChildSession = {
      sessionId: "self-resuming-session",
      prompt: () => new Promise<void>((resolve) => {
        resolvePrompt = resolve
        emit({ type: "agent_start" })
      }),
      steer: async () => "handled",
      followUp: async () => "handled",
      abort: async () => {},
      subscribe: (listener) => {
        listeners.add(listener)
        return () => listeners.delete(listener)
      },
      getLastAssistantText: () => lastText,
      dispose: () => {},
    }
    const end = (text: string): void => {
      lastText = text
      emit({ type: "message_end", message: { role: "assistant", content: [{ type: "text", text }], stopReason: "stop" } })
      emit({ type: "agent_end" })
    }
    const runner = createInProcessManagedRunner(new InProcessRunner({ createSession: async () => session }))
    const manager = createTaskManager({
      store,
      runners: { "in-process": runner, process: new FakeRunner() },
      planner: categoryPlanner(),
      config: settings(),
      cwd: project,
    })
    const started = await manager.start(baseSpec({ execution_mode: "in-process" }))
    if (started.kind !== "started") throw new Error("expected started")
    const taskId = started.task_id
    const waitForTerminal = () => manager.waitFor(taskId, { signal: AbortSignal.timeout(2_000) })
    try {
      const first = waitForTerminal()
      end("waiting for the monitor")
      emit({ type: "agent_idle" })
      resolvePrompt?.()
      expect((await first).status).toBe("completed")

      // when autonomous starts arrive, without handle.followUp or another parent prompt
      for (const [index, response] of ["monitor found the fix", "verification passed"].entries()) {
        const reopened = once(events, "task_self_resumed", { signal: AbortSignal.timeout(2_000) }).catch(() => {
          throw new Error(`waited for task_self_resumed; task stayed ${store.load(taskId)?.status}`)
        })
        emit({ type: "agent_start" })
        await reopened
        expect(store.load(taskId)?.status).toBe("running")
        expect(store.load(taskId)?.notification.run_epoch).toBe(index + 1)

        const terminal = waitForTerminal()
        end(response)
        expect(store.load(taskId)?.status).toBe("running")
        emit({ type: "agent_idle" })
        const settled = await terminal
        expect(settled.status).toBe("completed")
        expect(settled.final_response).toBe(response)
        expect(buildCompletionDetails(settled).resumed_turn).toBe(true)
      }

      // then an explicit revival still owns exactly one epoch and its ordinary completion
      await manager.continueTask(taskId, "one more thing")
      const terminal = waitForTerminal()
      end("asked for")
      emit({ type: "agent_idle" })
      resolvePrompt?.()
      const settled = await terminal
      expect(settled.notification.run_epoch).toBe(3)
      expect(settled.final_response).toBe("asked for")
      expect(buildCompletionDetails(settled).resumed_turn).toBeUndefined()
    } finally {
      const handle = manager.getResidentHandle(taskId)
      manager.forget(taskId)
      await handle?.dispose()
    }
  })

  test("#given a completed resident child #when its monitor wakes it #then the record runs again under the next epoch and its end is a second completion marked as a resumed turn", async () => {
    // given
    const { store, fake, taskId } = await startedChild()
    fake.settle({ status: "completed", finalResponse: "waiting for the test run monitor" })
    await flush()
    expect(store.load(taskId)?.status).toBe("completed")

    // when
    fake.selfResume()
    await flush()
    const resumed = store.load(taskId)
    fake.settle({ status: "completed", finalResponse: "tests passed, fixed the last failure" })
    await flush()

    // then
    expect(resumed?.status).toBe("running")
    expect(resumed?.notification.run_epoch).toBe(1)
    const settled = store.load(taskId)
    if (settled === null || settled === undefined) throw new Error("record vanished")
    expect(settled.status).toBe("completed")
    expect(settled.final_response).toBe("tests passed, fixed the last failure")
    expect(settled.notification.run_epoch).toBe(1)
    expect(buildCompletionDetails(settled).resumed_turn).toBe(true)
  })

  test("#given a child resumed on its own #when a task_send later revives it #then that revival's completion is not marked as a resumed turn", async () => {
    // given
    const { manager, store, fake, taskId } = await startedChild()
    fake.settle({ status: "completed", finalResponse: "first" })
    await flush()
    fake.selfResume()
    await flush()
    fake.settle({ status: "completed", finalResponse: "resumed" })
    await flush()

    // when
    await manager.continueTask(taskId, "one more thing")
    fake.settle({ status: "completed", finalResponse: "asked for" })
    await flush()

    // then
    const settled = store.load(taskId)
    if (settled === null || settled === undefined) throw new Error("record vanished")
    expect(settled.notification.run_epoch).toBe(2)
    expect(buildCompletionDetails(settled).resumed_turn).toBeUndefined()
  })

  test("#given a cancelled child #when a late self-resume fires #then the record stays cancelled", async () => {
    // given
    const { manager, store, fake, taskId } = await startedChild()
    await manager.cancelTask(taskId)
    await flush()

    // when
    fake.selfResume()
    await flush()

    // then
    expect(store.load(taskId)?.status).toBe("cancelled")
  })
})
