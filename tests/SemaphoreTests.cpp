#include "kernel/semaphore.h"
#include "libs/errno.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <thread>

namespace {

namespace Semaphore = Libs::LibKernel::Semaphore;
using Libs::LibKernel::KERNEL_ERROR_EACCES;
using Libs::LibKernel::KERNEL_ERROR_EBUSY;
using Libs::LibKernel::KERNEL_ERROR_ECANCELED;
using Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;
using Libs::LibKernel::KernelUseconds;

constexpr uint32_t       SEMA_ATTR_PRIORITY  = 0x02;
constexpr KernelUseconds LONG_TIMEOUT_MICROS = 10000000;

std::atomic_int  g_queued_waiters {0};
std::atomic_int  g_signal_polls {0};
thread_local int g_wait_priority = 700;

} // namespace

namespace Libs::LibKernel {

// Called with the semaphore mutex held, right before a priority waiter is queued.
int PthreadGetCurrentPriorityForKernel() {
	g_queued_waiters.fetch_add(1, std::memory_order_release);
	return g_wait_priority;
}

// Called after each condition variable wait of a blocked waiter, signaled or timed out.
void KernelDispatchPendingSignalForCurrentThread() {
	g_signal_polls.fetch_add(1, std::memory_order_relaxed);
}

int KYTY_SYSV_ABI KernelClockGettime(KernelClockid /*clock_id*/, KernelTimespec* /*tp*/) {
	std::abort();
}

} // namespace Libs::LibKernel

namespace {

struct Waiter {
	std::thread      thread;
	std::atomic_bool returned {false};
	KernelUseconds   timeout = LONG_TIMEOUT_MICROS;
	int              result  = OK;
};

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SemaphoreTests: failed: %s\n", text);
		std::abort();
	}
}

template <typename Pred>
void WaitUntil(Pred pred, const char* text) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
	while (!pred()) {
		Check(std::chrono::steady_clock::now() < deadline, text);
		std::this_thread::yield();
	}
}

Semaphore::KernelSema CreateSema(const char* name, int init, int max) {
	Semaphore::KernelSema sema = nullptr;
	Check(Semaphore::KernelCreateSema(&sema, name, SEMA_ATTR_PRIORITY, init, max, nullptr) == OK,
	      "create semaphore");
	return sema;
}

// Returns once the waiter is queued on the semaphore.
void StartWaiter(Waiter* waiter, Semaphore::KernelSema sema, int need, bool timed,
                 int priority = 700) {
	const int queued = g_queued_waiters.load(std::memory_order_acquire);
	waiter->thread   = std::thread([waiter, sema, need, timed, priority] {
		g_wait_priority = priority;
		auto* timeout   = (timed ? &waiter->timeout : nullptr);
		waiter->result  = Semaphore::KernelWaitSema(sema, need, timeout);
		waiter->returned.store(true, std::memory_order_release);
	});
	WaitUntil([queued] { return g_queued_waiters.load(std::memory_order_acquire) > queued; },
	          "waiter queued");
}

int CountReturned(std::span<const Waiter> waiters) {
	int count = 0;
	for (const auto& waiter: waiters) {
		count += (waiter.returned.load(std::memory_order_acquire) ? 1 : 0);
	}
	return count;
}

void JoinAll(std::span<Waiter> waiters) {
	WaitUntil([waiters] { return CountReturned(waiters) == static_cast<int>(waiters.size()); },
	          "all waiters return");
	for (auto& waiter: waiters) {
		waiter.thread.join();
	}
}

void TestSignalReleasesOneWaiter() {
	constexpr int WAITER_COUNT = 4;

	auto* sema = CreateSema("signal-one", 0, WAITER_COUNT);

	Waiter waiters[WAITER_COUNT];
	for (auto& waiter: waiters) {
		StartWaiter(&waiter, sema, 1, true);
	}

	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal one unit");
	WaitUntil([&] { return CountReturned(waiters) == 1; }, "signal releases a waiter");
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	Check(CountReturned(waiters) == 1, "signal releases exactly one waiter");

	Check(Semaphore::KernelSignalSema(sema, WAITER_COUNT - 1) == OK, "signal remaining units");
	JoinAll(waiters);
	for (const auto& waiter: waiters) {
		Check(waiter.result == OK, "signaled waiters succeed");
	}
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete signal-one semaphore");
}

void TestSignalWakesWaiterWhoseNeedFits() {
	auto* sema = CreateSema("need-fits", 0, 2);

	Waiter first;
	Waiter second;
	StartWaiter(&first, sema, 2, true);
	StartWaiter(&second, sema, 1, true);

	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal one unit past the queue head");
	WaitUntil([&] { return second.returned.load(std::memory_order_acquire); },
	          "signal releases the waiter whose need fits");
	Check(!first.returned.load(std::memory_order_acquire), "queue head keeps waiting");

	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal second unit");
	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal third unit");
	WaitUntil([&] { return first.returned.load(std::memory_order_acquire); },
	          "queue head returns once its count is available");
	first.thread.join();
	second.thread.join();
	Check(first.result == OK && second.result == OK, "both waiters succeed");
	Check(Semaphore::KernelPollSema(sema, 1) == KERNEL_ERROR_EBUSY,
	      "satisfied waiters consume their counts");
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete need-fits semaphore");
}

void TestSignalFollowsPriorityOrder() {
	constexpr int WAITER_COUNT = 4;

	auto* sema = CreateSema("priority", 0, WAITER_COUNT);

	// Lower values run first; equal priorities keep their arrival order.
	const int priorities[WAITER_COUNT] = {700, 256, 700, 480};
	const int wake_order[WAITER_COUNT] = {1, 3, 0, 2};
	Waiter    waiters[WAITER_COUNT];
	for (int i = 0; i < WAITER_COUNT; i++) {
		StartWaiter(&waiters[i], sema, 1, true, priorities[i]);
	}

	for (int i = 0; i < WAITER_COUNT; i++) {
		auto& expected = waiters[wake_order[i]];
		Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal one unit");
		WaitUntil([&] { return expected.returned.load(std::memory_order_acquire); },
		          "signal releases the next waiter in priority order");
		Check(CountReturned(waiters) == i + 1, "signal releases one waiter at a time");
	}

	JoinAll(waiters);
	for (const auto& waiter: waiters) {
		Check(waiter.result == OK, "priority waiters succeed");
	}
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete priority semaphore");
}

void TestCancelWakesAllWaiters() {
	constexpr int WAITER_COUNT = 3;

	auto* sema = CreateSema("cancel", 0, 4);

	Waiter waiters[WAITER_COUNT];
	StartWaiter(&waiters[0], sema, 1, false);
	StartWaiter(&waiters[1], sema, 2, true);
	StartWaiter(&waiters[2], sema, 4, false);

	int waiting = 0;
	Check(Semaphore::KernelCancelSema(sema, 3, &waiting) == OK, "cancel semaphore");
	Check(waiting == WAITER_COUNT, "cancel reports every waiter");
	JoinAll(waiters);
	for (const auto& waiter: waiters) {
		Check(waiter.result == KERNEL_ERROR_ECANCELED, "cancel releases every waiter");
	}
	Check(Semaphore::KernelPollSema(sema, 3) == OK, "cancel sets the new count");
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete cancel semaphore");
}

void TestDeleteWakesAllWaiters() {
	constexpr int WAITER_COUNT = 3;

	auto* sema = CreateSema("delete", 0, 2);

	Waiter waiters[WAITER_COUNT];
	StartWaiter(&waiters[0], sema, 1, false);
	StartWaiter(&waiters[1], sema, 1, true);
	StartWaiter(&waiters[2], sema, 2, false);

	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete semaphore with waiters");
	JoinAll(waiters);
	for (const auto& waiter: waiters) {
		Check(waiter.result == KERNEL_ERROR_EACCES, "delete releases every waiter");
	}
}

void TestTimedWaitExpires() {
	auto* sema = CreateSema("timeout", 0, 1);

	KernelUseconds timeout = 20000;
	const auto     start   = std::chrono::steady_clock::now();
	Check(Semaphore::KernelWaitSema(sema, 1, &timeout) == KERNEL_ERROR_ETIMEDOUT,
	      "timed wait expires");
	Check(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(10),
	      "timed wait does not return too early");
	Check(timeout == 0, "expired wait reports no time left");

	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal after timeout");
	Check(Semaphore::KernelPollSema(sema, 1) == OK, "expired waiter leaves the queue");
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete timeout semaphore");
}

void TestWaitPollsGuestSignals(bool timed) {
	auto* sema = CreateSema("signal-poll", 0, 1);

	g_signal_polls.store(0, std::memory_order_relaxed);
	Waiter waiter;
	StartWaiter(&waiter, sema, 1, timed);
	WaitUntil([] { return g_signal_polls.load(std::memory_order_relaxed) >= 2; },
	          "blocked wait keeps polling guest signals");
	Check(!waiter.returned.load(std::memory_order_acquire),
	      "signal polls do not release the waiter");

	Check(Semaphore::KernelSignalSema(sema, 1) == OK, "signal polling waiter");
	WaitUntil([&] { return waiter.returned.load(std::memory_order_acquire); },
	          "signal releases the polling waiter");
	waiter.thread.join();
	Check(waiter.result == OK, "polling waiter succeeds");
	Check(Semaphore::KernelDeleteSema(sema) == OK, "delete signal-poll semaphore");
}

} // namespace

int main() {
	TestSignalReleasesOneWaiter();
	TestSignalWakesWaiterWhoseNeedFits();
	TestSignalFollowsPriorityOrder();
	TestCancelWakesAllWaiters();
	TestDeleteWakesAllWaiters();
	TestTimedWaitExpires();
	TestWaitPollsGuestSignals(false);
	TestWaitPollsGuestSignals(true);
	std::printf("SemaphoreTests: all cases passed\n");
	return 0;
}
