from __future__ import annotations
from dataclasses import dataclass
import time

# 线程相关的标准库工具。
# Condition：条件变量，适合“等某个状态变化后再继续”的场景。
# Event：事件对象，适合在多个线程之间广播一个布尔状态，比如“停止运行”。
# Lock：互斥锁，适合保护共享变量，避免竞争条件。
# Thread：线程对象，用来创建和启动并发执行的线程。
from threading import Condition, Event, Lock, Thread

# 类型注解模块，用来说明参数和返回值的类型，提高可读性，也方便静态检查。
from typing import Any, Callable, Optional, Sequence

@dataclass(frozen=True)
class ObservationRecord:
	# step 表示这份观测对应的是第几步执行之后产生的。
	# data 保存实际观测数据，可以是字典、张量、状态对象等任意内容。
	step: int
	data: Any

@dataclass(frozen=True)
class ActionPlan:
	# plan_id 是动作计划的版本号，用来区分新旧计划。
	# source_obs_step 表示这份计划是基于哪一步观测推理出来的。
	# actions 保存动作序列，tuple 表示它是不可变的，适合在多个线程之间共享。
	plan_id: int
	source_obs_step: int
	actions: tuple[Any, ...]

# 下面三个别名是为了把“你需要自己实现的函数”的签名写清楚。
# ObserveFn：观测函数，不接收参数，返回一份观测。
ObserveFn = Callable[[], Any]
# ObserveProcessFn：观测处理函数，输入一份原始观测数据，只做处理，不需要返回值。
ObserveProcessFn = Callable[[Any], None]
# InferFn：推理函数，不接收参数，直接读取内部状态并输出未来 n 步动作序列。
InferFn = Callable[[], Sequence[Any]]
# ExecuteFn：动作执行函数，输入一个动作并把它真正执行到环境里。
ExecuteFn = Callable[[Any], None]

class DualThreadMachine:
	"""双线程前向执行框架。

	线程 A：等待观测更新，基于最新观测做推理，并生成动作序列。
	线程 B：先做一次初始观测，首次需要等待第一份动作计划；之后持续执行当前动作序列，
	如果没有新计划，就直接沿用旧计划继续执行；每执行一个动作后重新观测。

	latency=True 时，新的动作序列会按观测步对齐：
	若新序列来自 source_obs_step=s1，而当前已执行到 s2 步，
	则从新序列的第 max(0, s2 - s1) 个动作索引开始执行。
	"""

	def __init__(
		self,
		observe_fn: ObserveFn,
		process_observation_fn: ObserveProcessFn,
		infer_fn: InferFn,
		execute_fn: ExecuteFn,
		latency: bool = False,
	) -> None:
		# 把外部传进来的三个业务函数保存到对象里，后面两个线程都会调用它们。
		self.observe_fn = observe_fn
		self.process_observation_fn = process_observation_fn
		self.infer_fn = infer_fn
		self.execute_fn = execute_fn
		# latency 是开关：True 表示启用“按执行进度对齐动作序列”的逻辑。
		self.latency = latency

		# Lock 是互斥锁：同一时刻只允许一个线程进入 with self._lock 保护的区域。
		self._lock = Lock()
		# Condition 绑定到同一把锁上，用于等待“新观测到来”。
		# wait_for 会在条件不满足时自动释放锁并阻塞，被 notify_all 唤醒后再重新获取锁。
		self._obs_ready = Condition(self._lock)
		# 另一个条件变量，用于等待“新动作计划到来”。
		self._plan_ready = Condition(self._lock)
		# Event 是线程间共享的布尔信号；这里表示是否应该停止整个框架。
		self._stop_event = Event()

		# 观测缓冲序列；线程 B 只负责追加，线程 A 每次整批取出后处理并清空。
		self._observation_buffer: list[ObservationRecord] = []

		# 保存最新动作计划；None 表示当前还没有可执行计划。
		self._latest_plan: Optional[ActionPlan] = None
		# 动作计划版本号，每次发布新计划时递增 1。
		self._plan_version = 0

		# 已经实际执行了多少步动作，用于 latency 对齐。
		self._executed_steps = 0
		# 防止 start() 被重复调用，避免重复创建线程。
		self._started = False
		# 保存创建出来的线程对象，方便后续 join 等待它们结束。
		self._threads: list[Thread] = []

	def start(self) -> None:
		# 如果已经启动过，就直接返回。
		if self._started:
			return

		# 标记已经启动，后续再次调用 start() 会被忽略。
		self._started = True
		# 创建线程 A：负责等待新观测并做推理。
		thread_foward = Thread(target=self._thread_foward, name="thread-a-infer", daemon=True)
		# 创建线程 B：负责初始观测、执行动作、重新观测。
		thread_action = Thread(target=self._thread_action, name="thread-b-observe-execute", daemon=True)
		# 把线程对象保存起来。
		self._threads = [thread_foward, thread_action]

		# 启动两个线程；start() 会让线程进入调度队列，之后由系统决定何时真正运行。
		for thread in self._threads:
			thread.start()

	def stop(self) -> None:
		# set() 会把事件状态置为 True，表示外部请求整个系统停止。
		self._stop_event.set()
		# 由于线程可能正阻塞在 wait_for() 上，所以要主动唤醒它们。
		# 先拿锁，再通知，能保证共享状态和唤醒动作保持一致。
		with self._lock:
			# 唤醒所有等待“新观测”的线程，通常是线程 A。
			self._obs_ready.notify_all()
			# 唤醒所有等待“新计划”的线程，通常是线程 B。
			self._plan_ready.notify_all()

	def join(self, timeout: Optional[float] = None) -> None:
		# join() 会阻塞当前线程，直到子线程结束或达到超时时间。
		# 常见用法是主线程 start() 后再 join()，等待后台任务全部完成。
		for thread in self._threads:
			thread.join(timeout=timeout)

	def _publish_observation(self, observation_data: Any) -> None:
		# 线程 B 会在这里发布最新观测。
		# 写共享状态一定要加锁，否则线程 A 可能读取到不一致的数据。
		with self._lock:
			# 把当前执行步数和观测数据封装成一条完整的观测记录。
			observation = ObservationRecord(
				step=self._executed_steps,
				data=observation_data,
			)
			# 将新观测放入缓冲区，等待线程 A 下一次统一处理。
			self._observation_buffer.append(observation)
			# 唤醒等待新观测的线程 A。
			self._obs_ready.notify_all()

	def _publish_plan(self, source_obs_step: int, actions: Sequence[Any]) -> None:
		# 线程 A 会在这里发布最新动作计划。
		# source_obs_step 记录这份计划是基于哪一步观测算出来的。
		with self._lock:
			# plan_id 用版本号自增生成，方便识别先后顺序。
			self._latest_plan = ActionPlan(
				plan_id=self._plan_version + 1,
				source_obs_step=source_obs_step,
				# 转成 tuple 是为了让计划在共享时保持不可变。
				actions=tuple(actions),
			)
			# 计划版本号加 1，通知线程 B 有新动作可用。
			self._plan_version += 1
			self._plan_ready.notify_all()

	def _thread_foward(self) -> None:
		# 只要没有 stop 信号，线程 A 就持续工作。
		while not self._stop_event.is_set():
			with self._lock:
				self._obs_ready.wait_for(
					# 当出现停止信号，或者缓冲区里有新观测时，wait_for 才会返回。
					lambda: self._stop_event.is_set()
					or len(self._observation_buffer) > 0
				)
				# 如果已经要求停止，就直接结束线程。
				if self._stop_event.is_set():
					return

				# 一次性取出当前全部待处理观测，并立刻清空缓冲区，避免推理时再混入旧数据。
				pending_observations = self._observation_buffer
				self._observation_buffer = []

			# 如果本轮没有任何观测，直接跳过。
			if not pending_observations:
				continue

			# 逐个处理缓冲里的观测；处理函数只负责副作用，不返回结果。
			print('[buffer] buffer len :', len(pending_observations))
			for observation in pending_observations:
				self.process_observation_fn(observation.data)

			# 调用你自己实现的推理函数。
			# 推理函数自己读取内部状态并输出未来 n 步动作序列。
			actions = tuple(self.infer_fn())

			# 推理过程可能比较耗时，但这里不做“推理后版本校验”。
			# 也就是说，只要线程 A 已经拿到这份观测，就继续把这次推理做完；
			# 即使推理期间线程 B 又发布了新观测，这次推理结果也照样保留并发布。
			# 这样做的好处是不会因为观测更新太快而频繁丢弃推理结果。
			if self._stop_event.is_set():
				return

			# 直接把这份动作序列发布出去。
			# 这里使用本批观测中最新的 step，表示这份计划是基于当前批次最后一条观测得到的。
			self._publish_plan(pending_observations[-1].step, actions)

	def _thread_action(self) -> None:
		# 线程 B 启动后先做一次初始观测，给线程 A 提供第一份输入。
		initial_observation = self.observe_fn()
		# 把初始观测发布到共享状态中，并唤醒线程 A。
		self._publish_observation(initial_observation)

		# active_plan_version 记录当前正在执行的是哪一版动作计划。
		# 只有第一次启动时需要等待第一份计划；后面如果没有新计划，就继续执行旧计划。
		active_plan_version = 0
		# active_plan 保存当前正在执行的计划。
		active_plan: Optional[ActionPlan] = None
		# next_action_index 表示当前计划中准备执行到第几个动作。
		next_action_index = 0

		# 线程 B 的主循环：第一次等到第一份计划，之后持续执行当前计划；
		# 旧计划执行完且还没有新计划时，就暂停等待下一次切换。
		while not self._stop_event.is_set():
			# 第一次必须等待线程 A 发布第一份动作计划。
			if active_plan is None:
				with self._lock:
					self._plan_ready.wait_for(
						lambda: self._stop_event.is_set()
						or self._plan_version > active_plan_version
					)
					# 如果已经要求停止，就退出线程。
					if self._stop_event.is_set():
						return

					# 取出第一份计划、版本号以及当前已执行步数。
					# executed_steps 后面会用于 latency 对齐。
					plan = self._latest_plan
					plan_version = self._plan_version
					executed_steps = self._executed_steps

				# 如果没有计划或计划为空，就继续等待第一份有效计划。
				if plan is None or not plan.actions:
					active_plan_version = plan_version
					continue

				# latency 开关控制是否启用“按当前执行进度对齐动作序列”的逻辑。
				# max(0, executed_steps - plan.source_obs_step) 的含义是：
				# - 如果新计划和当前进度一致，就从头开始执行；
				# - 如果当前已经执行了更多步，就跳过前面已经过时的动作，从更靠后的动作开始执行。
				if self.latency:
					next_action_index = max(0, executed_steps - plan.source_obs_step)
				else:
					# latency=False 时，不做对齐，始终从动作序列开头执行。
					next_action_index = 0

				active_plan = plan
				active_plan_version = plan_version
			else:
				# 后续不再等待新计划；如果线程 A 发布了更新，就切换到新计划，否则继续执行旧计划。
				with self._lock:
					if self._plan_version > active_plan_version:
						plan = self._latest_plan
						plan_version = self._plan_version
						executed_steps = self._executed_steps

						if plan is not None and plan.actions:
							if self.latency:
								next_action_index = max(0, executed_steps - plan.source_obs_step)
							else:
								next_action_index = 0
							active_plan = plan
							active_plan_version = plan_version

			# 如果还没有可执行计划，就直接进入下一轮。
			if active_plan is None or not active_plan.actions:
				continue

			# 旧计划执行完以后，如果还没有新计划，就暂停，不再回绕重放。
			if next_action_index >= len(active_plan.actions):
				time.sleep(0.01)
				continue

			# 逐个执行当前计划中的动作。
			# 如果中途来了新计划，下面的版本号检查会把旧计划打断；否则就一直沿用当前计划。
			while next_action_index < len(active_plan.actions) and not self._stop_event.is_set():
				# 每执行一个动作前，先确认计划版本是否仍然有效。
				with self._lock:
					if self._plan_version != active_plan_version:
						# 有新计划插队，旧计划立即停止执行。
						break

				# 取出当前动作并交给你自己实现的执行函数。
				action = active_plan.actions[next_action_index]
				self.execute_fn(action)

				# 每执行完一个动作，就把已执行步数加 1。
				# 这个值会作为下一轮 latency 对齐的依据。
				with self._lock:
					self._executed_steps += 1

				# 执行动作后立刻重新观测。
				# 这样线程 A 才能基于最新状态继续推理下一轮动作。
				updated_observation = self.observe_fn()
				# 将新观测发布出去，线程 A 收到通知后会立刻处理。
				self._publish_observation(updated_observation)

				# 进入动作序列中的下一个位置。
				next_action_index += 1

# 只有直接运行这个文件时，才会执行下面的示例入口。
if __name__ == "__main__":

	obs_value = 0
	n_action = 20

	# 下面三个函数是占位模板，表示这些业务逻辑由你自己补实现。
	def observe_fn() -> dict[str, Any]:
		# 观测函数：从环境中读取状态，并返回一份观测字典。
		# 这里抛出 NotImplementedError，是为了提醒你必须替换成真实实现。
		# raise NotImplementedError("请自己实现观测函数")
		global obs_value
		obs_value += 1
		print("获取观测:", obs_value)
		return {"obs": obs_value}

	def process_observation_fn(observation_data: dict[str, Any]) -> None:
		# 观测处理函数：处理单条原始观测数据并更新内部状态，不返回结果。
		print("处理观测:", observation_data)

	def infer_fn() -> Sequence[Any]:
		# 推理函数：直接读取已经被处理过的内部状态，输出未来 n 步动作序列。
		# 返回值用 Sequence[Any] 表示，列表、元组等有序结构都可以。
		# raise NotImplementedError("请自己实现推理函数")
		infer_value = obs_value * 10
		print("推理动作，读取当前状态输出动作序列:", [f"action_{infer_value}_{i}" for i in range(n_action)])
		return [f"action_{infer_value}_{i}" for i in range(n_action)]
		

	def execute_fn(action: Any) -> None:
		# 动作执行函数：把单个动作真正施加到环境或系统中。
		# 返回 None 表示它主要依赖副作用，不需要返回结果。
		# raise NotImplementedError("请自己实现动作执行函数")
		print(f"执行动作: {action}")


	# 创建双线程控制器，把三个核心业务函数注入进去。
	# latency=True 表示启用动作序列对齐逻辑。
	controller = DualThreadMachine(
		observe_fn=observe_fn,
		process_observation_fn=process_observation_fn,
		infer_fn=infer_fn,
		execute_fn=execute_fn,
		latency=True,
	)
	controller.start()
	# 显式等待线程运行，避免主线程先结束导致你看不到输出。
	# 正常情况下这里会一直运行；如果按 Ctrl+C，会先 stop() 再 join()，让后台线程优雅退出。
	try:
		controller.join()
	except KeyboardInterrupt:
		print("收到 Ctrl+C，正在停止...")
		controller.stop()
		controller.join()