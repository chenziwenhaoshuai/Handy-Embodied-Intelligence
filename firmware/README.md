# ESP32-S3 + BMI160 LAN 3D Dashboard

ESP32-S3 reads the BMI160 over 4-wire SPI, converts the raw readings into the robot body frame, and serves a 3D attitude dashboard to PCs on the same LAN. It also runs the trained policy on-board at 50 Hz and drives two SG90 servos. Wi-Fi credentials live in `main/wifi_config.h` (git-ignored, copy from `main/wifi_config.example.h`), and a setup access point is provided as a fallback.

## SPI Wiring

| BMI160 | ESP32-S3 |
|---|---|
| 3V3 | 3V3 |
| GND | GND |
| SCL | GPIO4 (SPI SCLK) |
| SDA | GPIO5 (SPI MOSI) |
| SAO/SDO | GPIO7 (SPI MISO) |
| CS | GPIO6 (SPI CS) |

`VIN`, `INT1`, `INT2`, `OCS`, `SCX`, and `SDX` are not connected in this demo.

## Use

The firmware connects to the Wi-Fi configured in `main/wifi_config.h` at startup and opens the IP printed by the serial log, for example `http://192.168.1.50/`.

If the configured network is unavailable or cannot be connected after several attempts, the device opens a setup access point: **SSID `BMI160-Setup`**, password **`12345678`**. Connect to it, open `http://192.168.4.1/`, and enter the target Wi-Fi credentials. Credentials are saved in NVS and retained after restart.

The page renders a 3D breadboard. The ESP32 samples the IMU at 200 Hz and pushes attitude updates to the page at 20 Hz; the `/imu` JSON endpoint remains available as a fallback. Drag the scene to inspect the model.

Place the IMU in its desired standard-forward position, then click `复位`. The current attitude becomes the zero point and the 3D model returns to its standard orientation.

The page provides four board-side attitude modes: fusion, gyro-only, accelerometer-only, and accelerometer Roll/Pitch with gyroscope Yaw (the default). The `GX 反向`, `GY 反向`, and `GZ 反向` buttons invert the corresponding accelerometer and gyroscope axes when the sensor is mounted upside down or with a reversed axis direction. Changing an axis automatically requests a new zero point.

The orientation filter uses accelerometer correction for roll/pitch and gyro integration for yaw. BMI160 has no magnetometer in this configuration, so yaw will drift gradually; this is expected.

Two 180-degree servos are available from the page: GPIO8 controls the first servo and GPIO9 controls the second, each with a 0-180 degree slider. Connect each servo's signal wire to its GPIO, connect servo power to a suitable external 5V supply, and connect that supply ground to ESP32 GND. Do not power the servos from the ESP32 3V3 pin.

Slider commands are coalesced in the browser so only the latest angle is sent while a previous request is in flight. This keeps rapid dragging from filling the ESP32 HTTP request queue.

## Build

```powershell
idf.py build
idf.py -p COM5 flash monitor
```
