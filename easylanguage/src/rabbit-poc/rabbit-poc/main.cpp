#include "rabbit-stock.h"

int main() {
	for (int i = 0; i < 3000; i++) {
		_backtest_message(
			"192.168.11.30", 5672, "guest", "guest",
			"TEST", "2019/09/18 11:10:00", "rabbit-poc",
			0, 0, 0, 0, 0
		);
	}
	return 0;
}