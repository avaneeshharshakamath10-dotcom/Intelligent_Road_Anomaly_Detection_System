#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <RF24.h>
#include <printf.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

/************************************************
 * OLED CONFIGURATION
 ************************************************/
#define SCREEN_WIDTH   128
#define SCREEN_HEIGHT   64

#define OLED_SDA       21
#define OLED_SCL       22

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

/************************************************
 * NRF24 CONFIGURATION
 ************************************************/
#define CE_PIN   4
#define CSN_PIN  5

// TEMPORARY REVERT for testing: back to the 2-argument constructor, since radio.begin()
// started failing immediately after adding the explicit 1MHz SPI speed argument. This
// isolates whether that specific change is incompatible with this RF24 library version
// when combined with the manual SPI.begin(18,19,23,CSN_PIN) call used for ESP32's custom pins.
RF24 radio(CE_PIN, CSN_PIN);

// SAME ADDRESS AS STM32
uint8_t address[5] = {'N','O','D','E','1'};

/************************************************
 * PACKET STRUCTURE
 * EXACTLY SAME AS STM32
 ************************************************/
typedef struct __attribute__((packed))
{
    uint8_t systemToken;
    uint8_t vehicleID;
    uint8_t alertCode;
    uint8_t frameCounter;
    uint8_t checksum;
    uint8_t reserved;

} VANET_Packet;

VANET_Packet rxPacket;

/************************************************
 * DISPLAY VARIABLES
 ************************************************/
bool alertActive = false;
unsigned long alertTime = 0;

const uint32_t DISPLAY_TIMEOUT = 3000;

/************************************************
 * CHECKSUM
 ************************************************/
uint8_t CalculateChecksum(VANET_Packet *pkt)
{
    return pkt->systemToken ^
           pkt->vehicleID ^
           pkt->alertCode ^
           pkt->frameCounter ^
           pkt->reserved;
}

/************************************************
 * OLED HOME SCREEN
 ************************************************/
void ShowWaitingScreen()
{
    display.clearDisplay();

    display.setTextSize(2);
    display.setCursor(12,0);
    display.println("V2V NODE");

    display.drawLine(0,20,128,20,SSD1306_WHITE);

    display.setTextSize(1);

    display.setCursor(0,30);
    display.print("Channel : 80");

    display.setCursor(0,42);
    display.print("Address : NODE1");

    display.setCursor(0,54);
    display.print("Waiting...");

    display.display();
}

/************************************************
 * SETUP
 ************************************************/
void setup()
{
    Serial.begin(115200);
    delay(1000);

    printf_begin();

    Serial.println();
    Serial.println("================================");
    Serial.println("ESP32 V2V RECEIVER");
    Serial.println("================================");

    /************* OLED *************/
    Wire.begin(OLED_SDA, OLED_SCL);

    if(!display.begin(SSD1306_SWITCHCAPVCC,0x3C))
    {
        Serial.println("OLED FAILED");

        while(1);
    }

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    display.setTextSize(2);
    display.setCursor(8,20);
    display.println("BOOTING");

    display.display();

    delay(1000);

    /************* SPI *************/
    SPI.begin(18,19,23,CSN_PIN);

    // Slow the SPI clock to 1MHz (down from the default ~10-40MHz) for better signal
    // integrity on breadboard wiring -- set here at the SPI peripheral level instead of
    // through the RF24 constructor, since the 3-argument constructor wasn't compatible
    // with this library version when combined with a manual SPI.begin() call.
    SPI.setFrequency(1000000);

    /************* NRF24 *************/
    if(!radio.begin())
    {
        Serial.println("NRF24 NOT FOUND");

        display.clearDisplay();
        display.setTextSize(2);
        display.setCursor(0,20);
        display.println("NRF ERROR");
        display.display();

        while(1);
    }

    radio.setChannel(80);
    radio.setDataRate(RF24_1MBPS);
    radio.setPALevel(RF24_PA_HIGH);

    radio.setPayloadSize(sizeof(VANET_Packet));

    radio.setCRCLength(RF24_CRC_8);

    // Broadcast mode: Auto-Ack off. If multiple vehicle-node ESP32s are all listening
    // on this same address/channel, they must NOT all try to acknowledge the same
    // packet at once -- that would collide over the air. This node just listens and
    // displays; it never needs to confirm receipt back to the sender.
    radio.setAutoAck(false);

    radio.openReadingPipe(0,address);

    radio.startListening();

    // Explicit, forceful clear -- the library's own internal flush during begin()/
    // startListening() doesn't always reliably land given this link's signal integrity
    // history. This directly flushes the RX FIFO and clears RX_DR/TX_DS/MAX_RT so we
    // start from a guaranteed-clean state rather than trusting an implicit one.
    radio.flush_rx();
    radio.flush_tx();
    bool rxdr, txds, maxrt;
    radio.whatHappened(rxdr, txds, maxrt); // reading clears these flags as a side effect

    Serial.println("NRF24 INITIALIZED");
    Serial.println();

    radio.printDetails();

    ShowWaitingScreen();
}
/************************************************
 * MAIN LOOP
 ************************************************/
void loop()
{
    if (radio.available())
    {
        while (radio.available())
        {
            radio.read(&rxPacket, sizeof(VANET_Packet));
        }

        Serial.println("--------------------------------");
        Serial.println("PACKET RECEIVED");

        Serial.print("Token      : 0x");
        Serial.println(rxPacket.systemToken, HEX);

        Serial.print("Vehicle ID : ");
        Serial.println(rxPacket.vehicleID);

        Serial.print("Alert Code : ");
        Serial.println(rxPacket.alertCode);

        Serial.print("Frame      : ");
        Serial.println(rxPacket.frameCounter);

        Serial.print("Checksum   : 0x");
        Serial.println(rxPacket.checksum, HEX);

        uint8_t checksum = CalculateChecksum(&rxPacket);

        if (rxPacket.systemToken != 0xAA)
        {
            Serial.println("INVALID TOKEN");
            // Defensive re-flush: if this keeps happening back-to-back, the RX_DR flag
            // or FIFO may be stuck rather than genuinely receiving repeated garbage.
            // Forcing a flush here breaks that loop instead of spinning forever.
            radio.flush_rx();
            return;
        }

        if (checksum != rxPacket.checksum)
        {
            Serial.println("CHECKSUM ERROR");
            return;
        }

        Serial.println("PACKET VERIFIED");

        alertActive = true;
        alertTime = millis();

        display.clearDisplay();

        display.drawRect(0,0,128,64,SSD1306_WHITE);
        display.drawRect(2,2,124,60,SSD1306_WHITE);

        display.setTextSize(1);

        display.setCursor(8,8);
        display.print("Vehicle ");
        display.print(rxPacket.vehicleID);

        display.setCursor(80,8);
        display.print("F:");
        display.print(rxPacket.frameCounter);

        display.drawLine(6,20,122,20,SSD1306_WHITE);

        switch(rxPacket.alertCode)
        {
            case 1:

                // "POTHOLE AHEAD" is 13 chars; at text size 2 (12px/char) that's ~156px,
                // wider than the 128px screen, so it's split across two centered lines.
                display.setTextSize(2);
                display.setCursor(14,26);
                display.println("POTHOLE");
                display.setCursor(20,46);
                display.println("AHEAD");

                Serial.println("*** POTHOLE AHEAD ***");

                break;

            case 2:

                display.setTextSize(2);
                display.setCursor(26,26);
                display.println("HUMP");
                display.setCursor(26,46);
                display.println("AHEAD");

                Serial.println("*** HUMP AHEAD ***");

                break;

            default:

                display.setTextSize(1);
                display.setCursor(18,32);
                display.println("UNKNOWN ALERT");

                Serial.println("UNKNOWN ALERT");

                break;
        }

        display.display();
    }

    /*******************************************
     * Return to waiting screen after 3 seconds
     *******************************************/
    if (alertActive)
    {
        if (millis() - alertTime > DISPLAY_TIMEOUT)
        {
            alertActive = false;

            ShowWaitingScreen();

            Serial.println("Waiting for next packet...");
        }
    }
}
