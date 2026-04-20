import time


class _LedBit:
    def __init__(self, bank, bit_index):
        self._bank = bank
        self._mask = 1 << bit_index

    def on(self):
        current = self._bank.read(0)
        self._bank.write(0, current | self._mask)

    def off(self):
        current = self._bank.read(0)
        self._bank.write(0, current & ~self._mask)


class _BtnBit:
    def __init__(self, bank, bit_index):
        self._bank = bank
        self._mask = 1 << bit_index

    def read(self):
        return 1 if (self._bank.read(0) & self._mask) else 0


class IOHandler:
    def __init__(self, overlay):
        self._leds_bank = overlay.leds_gpio
        self._last_led_pattern = None
        leds_bank = overlay.leds_gpio
        btns_bank = overlay.btns_gpio

        self.leds = [_LedBit(leds_bank, index) for index in range(4)]
        self.btn0 = _BtnBit(btns_bank, 0)
        self.btn1 = _BtnBit(btns_bank, 1)
        self.btn2 = _BtnBit(btns_bank, 2)
        self.btn3 = _BtnBit(btns_bank, 3)
        self.switches_gpio = getattr(overlay, "switches_gpio", None)
        self.rgbleds_gpio = getattr(overlay, "rgbleds_gpio", None)

        self._write_led_pattern(0)

    def read_buttons(self):
        return {
            "btn0": self.btn0.read(),
            "btn1": self.btn1.read(),
            "btn2": self.btn2.read(),
            "btn3": self.btn3.read(),
        }

    def set_led_countdown(self, remaining, total):
        active = int((remaining / total) * len(self.leds))
        self._write_led_pattern((1 << active) - 1 if active > 0 else 0)

    def show_player_select(self, selected_players, flash_all):
        selected_players = max(1, min(len(self.leds), int(selected_players)))
        if flash_all:
            pattern = (1 << len(self.leds)) - 1
        else:
            pattern = (1 << selected_players) - 1
        self._write_led_pattern(pattern)

    def show_result(self, success):
        for _ in range(3):
            for led in self.leds:
                if success:
                    led.on()
                else:
                    led.off()
            time.sleep(0.2)
            for led in self.leds:
                if success:
                    led.off()
                else:
                    led.on()
            time.sleep(0.2)

    def clear_leds(self):
        self._write_led_pattern(0)

    def _write_led_pattern(self, pattern):
        if pattern == self._last_led_pattern:
            return
        self._leds_bank.write(0, pattern)
        self._last_led_pattern = pattern
