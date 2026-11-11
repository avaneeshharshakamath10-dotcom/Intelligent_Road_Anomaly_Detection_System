# Intelligent_Road_Anomaly_Detection_System


A low-cost road-safety prototype. A camera pipeline detects **potholes and speed humps**, estimates how far away they are, decides what the driver should do, and **broadcasts a wireless alert** that a second node (ESP32 with an OLED) displays as **"POTHOLE AHEAD"** or **"HUMP AHEAD"**.

**Stack:** YOLOv8s · ByteTrack · MiDaS (virtual LiDAR) · decision-level fusion · STM32 Nucleo-L432KC · nRF24L01 · ESP32 · SSD1306 OLED

> **Disclaimer:** this is an academic prototype for research and learning. It is **not** a certified automotive safety system and must not replace driver judgement. The wireless link demonstrates the *broadcast pattern* of V2V hazard warning; it does **not** implement the DSRC / C-V2X standards (see [Limitations](#limitations)).

---

## Table of contents
1. [How it works](#how-it-works)
2. [Features](#features)
3. [Hardware](#hardware)
4. [Wiring](#wiring)
5. [Software requirements](#software-requirements)
6. [Getting started](#getting-started)
7. [Radio and packet format](#radio-and-packet-format)
8. [Dashboard](#dashboard)
9. [Verifying the radio](#verifying-the-radio)
10. [Troubleshooting and lessons learned](#troubleshooting-and-lessons-learned)
11. [Results](#results)
12. [Limitations](#limitations)
13. [Future work](#future-work)
14. [Repository structure](#repository-structure)
15. [Team, references, license](#team-references-license)

---

## How it works

```mermaid
flowchart LR
  A[Road video] --> B[YOLOv8s + ByteTrack<br/>detect and track]
  A --> C[MiDaS<br/>depth map]
  B --> D[Decision-level fusion<br/>lane, distance, threat level]
  C --> D
  D --> E[Cockpit dashboard<br/>+ LiDAR window]
  D --> F[UART command<br/>POTHOLE / HUMP]
  F --> G[STM32 Nucleo-L432KC<br/>builds 6-byte packet]
  G --> H[nRF24L01 TX]
  H -. 2.4 GHz broadcast, channel 80 .-> I[nRF24L01 RX]
  I --> J[ESP32<br/>validates packet]
  J --> K[SSD1306 OLED<br/>POTHOLE / HUMP AHEAD]
```

1. **Detection:** a custom-trained YOLOv8s model finds potholes and humps; ByteTrack gives each hazard a persistent ID so it is shown and counted once.
2. **Depth:** one camera cannot measure range, so MiDaS monocular depth acts as a "virtual LiDAR".
3. **Fusion:** the fusion engine blends a position-in-frame estimate (60%) with the MiDaS-derived estimate (40%), assigns a lane, and classifies each hazard as `EMERGENCY_BRAKE`, `SLOW_DOWN` or `MONITOR`, with hysteresis so the alert does not flicker.
4. **Alert:** when action is needed, Python sends `POTHOLE\r\n` or `HUMP\r\n` over USB serial to the STM32.
5. **Broadcast:** the STM32 builds a 6-byte packet and transmits it three times through an nRF24L01 with Auto-Acknowledge disabled, so any receiver on the same channel and address can capture it.
6. **Display:** the ESP32 checks the token and checksum and shows the alert on the OLED for 3 seconds.

## Features

- Custom dataset: about 1,000 annotated images (potholes, humps, normal road), augmented to about 2,000; YOLOv8s and YOLOv8n trained for 50 epochs on Google Colab (YOLOv8s deployed)
- Persistent hazard IDs with ByteTrack and a per-session **pothole / hump counter**
- Fused distance estimate, lane assignment, hysteresis latch and a 15-frame blind-spot memory
- Cockpit-style dashboard with a **recommended driver action**: `CLEAR ROAD`, `BRAKE`, `SLOW DOWN`, `TURN LEFT`, `TURN RIGHT`
- Separate **virtual LiDAR window** that shows the depth map only inside tracked hazard boxes
- STM32 firmware with interrupt-driven UART, packet builder and an nRF24L01 driver (verified register writes)
- Broadcast receiver on ESP32 with token and checksum validation

## Hardware

| Component | Purpose |
|---|---|
| Laptop / PC (Windows) | Runs the perception pipeline and dashboard |
| STM32 Nucleo-L432KC | UART bridge, packet builder, nRF24L01 driver |
| nRF24L01 + adapter (x2) | 2.4 GHz transmitter and receiver |
| ESP32 DevKit | Receiver controller |
| SSD1306 OLED 128x64 (I2C, address 0x3C) | Alert display |
| 100 uF + 10 uF capacitors (x2 sets) | Decoupling across VCC/GND at each radio adapter |
| Jumper wires, USB cables | Direct wiring (see lessons learned) |

## Wiring

### STM32 Nucleo-L432KC to nRF24L01

| Signal | STM32 pin | Nucleo header | nRF24L01 |
|---|---|---|---|
| SCK | PB3 | D13 | SCK |
| MISO | PB4 | D12 | MISO |
| MOSI | PB5 | D11 | MOSI |
| CE | PB0 | D3 | CE |
| CSN | PB1 | D6 | CSN |
| VCC | 5V | 5V | VCC (adapter with onboard regulator) |
| GND | GND | GND | GND |

The USB cable to the laptop provides power and the virtual COM port (USART2, 115200 baud).



### ESP32 to nRF24L01 and OLED

| Signal | ESP32 pin | Device pin |
|---|---|---|
| SCK / MISO / MOSI | GPIO 18 / 19 / 23 | nRF24 SCK / MISO / MOSI |
| CE / CSN | GPIO 4 / 5 | nRF24 CE / CSN |
| VCC | 5V (VIN) | nRF24 adapter VCC (with capacitors) |
| SDA / SCL | GPIO 21 / 22 | OLED SDA / SCL |
| VDD | 3V3 | OLED VDD |



> **Power warning:** the nRF24L01 chip is 3.3 V only. Feeding **5 V** is correct **only** for adapters that have an onboard regulator. If your adapter has no regulator, use 3.3 V.

## Software requirements

| Part | Tools |
|---|---|
| Perception (Python) | Python 3, `opencv-python`, `torch`, `torchvision`, `ultralytics`, `numpy`, `pyserial` |
| Depth model | MiDaS_small, downloaded automatically through `torch.hub` on first run (internet needed once) |
| STM32 firmware | STM32CubeIDE with the STM32 HAL for the L4 series |
| ESP32 firmware | Arduino IDE, ESP32 board package, libraries **RF24**, **Adafruit GFX**, **Adafruit SSD1306** |
| Training | Google Colab (GPU), Ultralytics |

## Getting started

### 1. Perception pipeline (Python)

```bash
pip install opencv-python torch torchvision ultralytics numpy pyserial
```

1. Put your trained weights (for example `road_small.pt`) and a road video (for example `road_test.mp4`) next to `main.py`, or edit the paths in `config.py`:
   ```python
   video_path: str = "road_test.mp4"
   yolo_weights: str = "road_small.pt"
   ```
2. Set the STM32's serial port in `main.py` (`STM32_PORT = "COM3"`). Check the port in Windows Device Manager ("STMicroelectronics STLink Virtual COM Port").
3. Run:
   ```bash
   python main.py
   ```
   Press **q** to quit. If the serial port cannot be opened, the program continues in simulation mode (dashboard only).

Key settings in `config.py`: confidence 0.70, IoU 0.45, inference every 2nd frame, UART resend interval 1.5 s.

### 2. STM32 firmware

1. Open the project in STM32CubeIDE, build and flash it to the Nucleo-L432KC.
2. Open a serial terminal (for example PuTTY) on the Nucleo's COM port at **115200 baud, 8N1**.
3. Reset the board. You should see the boot register dump see [Verifying the radio] and `Waiting for command...`.
4. Test without Python: type `POTHOLE` and press Enter. The STM32 prints `Sending POTHOLE! ID=1 ...` and transmits.

> Close the terminal before running `main.py`: only one program can hold a COM port.

### 3. ESP32 receiver

1. Open `esp32_receiver/esp32_receiver.ino` in the Arduino IDE (the `.ino` file must have the same name as its folder, with no other `.ino` files in that folder).
2. Select your ESP32 board, upload, and open the Serial Monitor at **115200 baud**.
3. The OLED shows `V2V NODE / Channel : 80 / Address : NODE1 / Waiting...`. When a valid packet arrives it shows **POTHOLE AHEAD** or **HUMP AHEAD** for 3 seconds.

## Radio and packet format

| Setting | Value |
|---|---|
| Frequency | Channel 80 (2.480 GHz) |
| Data rate / power | 1 Mbps, 0 dBm |
| Address | `NODE1` (5 bytes), pipe 0 |
| CRC | 1 byte |
| Auto-Acknowledge | **Disabled** (broadcast to any number of receivers) |
| Redundancy | Each alert is sent **3 times**, 5 ms apart |

**6-byte packet** (packed struct, identical in STM32 and ESP32 code):

| Byte | Field | Value |
|---|---|---|
| 0 | `systemToken` | `0xAA` |
| 1 | `vehicleID` | `0x01` |
| 2 | `alertCode` | `1` = pothole, `2` = hump |
| 3 | `frameCounter` | increments per alert |
| 4 | `checksum` | XOR of bytes 0, 1, 2, 3 and 5 |
| 5 | `reserved` | `0x00` |

The checksum detects accidental corruption only. It is **not** encryption or authentication.

## Dashboard

- **Window 1:** annotated road feed (corner brackets, ID, class, distance, state) and a telemetry panel with the session hazard counter, fused range, recommended driver action and a rolling detection log. A flashing bar appears for `EMERGENCY_BRAKE`.
- **Window 2:** virtual LiDAR view (MiDaS depth shown only inside tracked boxes).

| Condition (centre lane unless stated) | State | Driver action |
|---|---|---|
| Distance <= 4.5 m, or box bottom in lowest 20% of frame | `EMERGENCY_BRAKE` | BRAKE (or TURN LEFT / RIGHT if the hazard is in a side lane) |
| Distance <= 9.5 m | `SLOW_DOWN` | SLOW DOWN |
| Otherwise, or hazard in a side lane | `MONITOR` | CLEAR ROAD |

A hysteresis latch keeps a high-threat state for a hazard that was already critical and is within 6 m.

## Verifying the radio

After flashing the STM32, the boot dump in the serial terminal should read:

```
CONFIG     = 0x0A
EN_AA      = 0x00
EN_RXADDR  = 0x01
SETUP_AW   = 0x03
SETUP_RETR = 0x00
RF_CH      = 0x50
RF_SETUP   = 0x06
STATUS     = 0x0E
```

Reset several times: the dump must be **identical every time**. Different values between reads usually mean a power or wiring problem, not a software bug. The `tools/` folder contains two raw-SPI test sketches for the ESP32 side (`nrf_spi_test`, `nrf_spi_test2`) that read the radio's registers without the RF24 library.

## Troubleshooting and lessons learned

| Symptom | Likely cause | Fix |
|---|---|---|
| STM32 never forwards alerts | Command not terminated, or wrong text | Send exactly `POTHOLE` or `HUMP` followed by CR LF |
| Random symbols in the terminal | Wrong baud rate or COM port | 115200 baud, correct ST-LINK COM port |
| `CONFIG` reads `0x08` (PWR_UP bit missing) | Corrupted write on a marginal supply | 100 ms power-up delay, verify and retry (already in `nrf24.c`) |
| All registers read `0x00`, adapter LED off | No power to the adapter | Check VCC and GND wiring |
| Scrambled or changing register values | Supply noise, regulator without headroom, breadboard contacts | Feed the adapter from 5 V, add 100 uF + 10 uF capacitors, use direct jumper wires |
| `RF_CH` reads 160 instead of 80 | SPI timing margin | Lower the SPI clock (500 kHz on STM32, 1 MHz on ESP32) |
| Nothing works despite correct wiring | Nucleo-32 header labels differ from MCU pin names | Use the D13/D12/D11/D3/D6 mapping in [Wiring]|
| ESP32 shows `NRF ERROR` | Radio not answering on SPI | Power-cycle (unplug USB), reseat the module in its adapter, try another cable; run `nrf_spi_test` |
| ESP32 `NRF ERROR` after editing the sketch | The 3-argument `RF24` constructor was not compatible with the manual `SPI.begin()` here | Keep `RF24 radio(CE_PIN, CSN_PIN)` and use `SPI.setFrequency(1000000)` |
| Some alerts not received | No hardware retransmission when Auto-Ack is off | Already mitigated by sending each alert 3 times |

## Results

> Fill these in from your own measurements. Do not publish numbers you have not measured.

| Measure | Result |
|---|---|
| YOLOv8s precision / recall / mAP@0.5 | TBD |
| YOLOv8n precision / recall / mAP@0.5 | TBD |
| Distance error (fused vs measured, mean absolute) | TBD |
| Alerts received / sent, conditions | TBD |
| Command-to-OLED latency | TBD (estimated from firmware timing: roughly 40-50 ms) |
| Tested radio range | TBD |

## Limitations

- **Not real V2V.** It uses the shared 2.4 GHz ISM band (not 5.9 GHz DSRC or C-V2X), a custom 6-byte packet (not SAE J2735), no GPS position, one transmitting node and one or more receiving nodes (not a peer mesh), and no message authentication (IEEE 1609.2-style).
- **Short range.** A basic nRF24L01 reaches tens of metres, which gives only a couple of seconds of warning at road speeds.
- **Approximate distance.** MiDaS gives *relative* depth. The fused distance uses tuned constants (25 and 150), not camera calibration, so treat it as an estimate.
- **Rule-based fusion.** The agreement flag is informational and does not gate the braking decision. If MiDaS fails to load, fusion returns no hazards instead of falling back to camera-only.
- **Dataset size.** About 2,000 images will under-represent night, rain and unusual road surfaces.
- **Prototype hardware.** Jumper-wire builds are sensitive to supply noise and loose contacts.

## Future work

- Add a GPS module so every packet carries the hazard's position and time
- Make every node both a transmitter and a receiver
- Add a message authentication code to prevent forged alerts
- Calibrate the camera and use the pinhole ground-plane model already implemented in `depth_estimator.py`
- Camera-only fallback when the depth model is unavailable
- Larger and more varied dataset; faster inference on a GPU
- Longer-range radio (LoRa, or DSRC / C-V2X hardware) and field trials



**References**
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- R. Ranftl et al., "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer," IEEE TPAMI, 2022 (MiDaS)
- Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV 2022
- Nordic Semiconductor, nRF24L01+ Product Specification
- STMicroelectronics, UM1956: STM32 Nucleo-32 boards user manual


