#include "nrf24.h"

/*-----------------------------------------------------------
    Read Register
-----------------------------------------------------------*/
uint8_t NRF24_ReadRegister(uint8_t reg)
{
    uint8_t command = R_REGISTER | (REGISTER_MASK & reg);
    uint8_t value = 0;

    NRF_CSN_LOW();

    HAL_SPI_Transmit(&hspi1, &command, 1, HAL_MAX_DELAY);
    HAL_SPI_Receive(&hspi1, &value, 1, HAL_MAX_DELAY);

    NRF_CSN_HIGH();

    return value;
}

/*-----------------------------------------------------------
    Write Register
-----------------------------------------------------------*/
void NRF24_WriteRegister(uint8_t reg, uint8_t value)
{
    uint8_t tx[2];

    tx[0] = W_REGISTER | (REGISTER_MASK & reg);
    tx[1] = value;

    NRF_CSN_LOW();

    HAL_SPI_Transmit(&hspi1, tx, 2, HAL_MAX_DELAY);

    NRF_CSN_HIGH();
}
/*-----------------------------------------------------------
    Write Buffer
-----------------------------------------------------------*/
void NRF24_WriteBuffer(uint8_t reg, uint8_t *data, uint8_t length)
{
    uint8_t command = W_REGISTER | reg;

    NRF_CSN_LOW();

    HAL_SPI_Transmit(&hspi1, &command, 1, HAL_MAX_DELAY);
    HAL_SPI_Transmit(&hspi1, data, length, HAL_MAX_DELAY);

    NRF_CSN_HIGH();
}

/*-----------------------------------------------------------
    Flush TX FIFO
-----------------------------------------------------------*/
void NRF24_FlushTX(void)
{
    uint8_t cmd = FLUSH_TX;

    NRF_CSN_LOW();
    HAL_SPI_Transmit(&hspi1, &cmd, 1, HAL_MAX_DELAY);
    NRF_CSN_HIGH();
}

/*-----------------------------------------------------------
    Flush RX FIFO
-----------------------------------------------------------*/
void NRF24_FlushRX(void)
{
    uint8_t cmd = FLUSH_RX;

    NRF_CSN_LOW();
    HAL_SPI_Transmit(&hspi1, &cmd, 1, HAL_MAX_DELAY);
    NRF_CSN_HIGH();
}

/*-----------------------------------------------------------
    Read STATUS
-----------------------------------------------------------*/
uint8_t NRF24_GetStatus(void)
{
    uint8_t tx = NOP;
    uint8_t rx = 0;

    NRF_CSN_LOW();

    HAL_SPI_TransmitReceive(&hspi1, &tx, &rx, 1, HAL_MAX_DELAY);

    NRF_CSN_HIGH();

    return rx;
}
/*-----------------------------------------------------------
    Initialise nRF24
-----------------------------------------------------------*/
void NRF24_Init(void)
{
    NRF_CE_LOW();

    /* nRF24 needs its power rail to fully stabilize before SPI writes are reliable.
       10ms was too tight, especially on a marginal/rewired power connection --
       bumped to 100ms per datasheet guidance. */
    HAL_Delay(100);

    /* CONFIG
       CRC Enabled
       Power Up
       TX Mode
    */
    NRF24_WriteRegister(CONFIG,
                        EN_CRC |
                        PWR_UP);

    /* Verify PWR_UP actually took. If the power rail is noisy, this single write is the
       most likely one to get corrupted (as seen: readback showed CONFIG=0x08, missing
       the PWR_UP bit). Retry a few times with a short settle delay rather than silently
       running with a radio that can never transmit or receive. */
    for (uint8_t attempt = 0; attempt < 5; attempt++)
    {
        uint8_t cfg = NRF24_ReadRegister(CONFIG);
        if ((cfg & PWR_UP) != 0)
        {
            break;
        }
        HAL_Delay(20);
        NRF24_WriteRegister(CONFIG, EN_CRC | PWR_UP);
    }

    /* After PWR_UP, nRF24 needs ~1.5ms before it can accept further commands */
    HAL_Delay(5);

    /* Disable Auto-Ack: this is now a broadcast link, not a paired point-to-point link.
       Any number of listening nRF24 nodes (same address + channel) can receive the
       same packet at once. If Auto-Ack stayed on, every listener would try to send
       an acknowledgment back simultaneously and their replies would collide over the air. */
    NRF24_WriteRegister(EN_AA,0x00);

    /* Enable Pipe0 */
    NRF24_WriteRegister(EN_RXADDR,0x01);

    /* 5 byte address */
    NRF24_WriteRegister(SETUP_AW,0x03);

    /* Auto Retransmit: irrelevant with Auto-Ack disabled (no ACK to wait for), left
       at 0 for clarity rather than a stale non-functional value. */
    NRF24_WriteRegister(SETUP_RETR,0x00);

    /* Channel 80 = 2.480GHz */
    NRF24_WriteRegister(RF_CH,80);

    /* 1Mbps
       0dBm
    */
    NRF24_WriteRegister(RF_SETUP,0x06);

    /* Clear Interrupts */
    NRF24_WriteRegister(STATUS,0x70);

    NRF24_FlushTX();
    NRF24_FlushRX();

    HAL_Delay(100);
}

/*-----------------------------------------------------------
    Send Packet
-----------------------------------------------------------*/
void NRF24_Send(uint8_t *data, uint8_t length)
{
    uint8_t cmd;
    uint8_t status;

    // 1. Force clear any lingering SPI hardware errors before communicating
    __HAL_SPI_CLEAR_OVRFLAG(&hspi1);

    NRF24_FlushTX();

    cmd = W_TX_PAYLOAD;

    NRF_CSN_LOW();
    HAL_SPI_Transmit(&hspi1, &cmd, 1, HAL_MAX_DELAY);
    HAL_SPI_Transmit(&hspi1, data, length, HAL_MAX_DELAY);
    NRF_CSN_HIGH();

    // 2. Pulse CE to execute transmission
    NRF_CE_HIGH();
    HAL_Delay(1);
    NRF_CE_LOW();

    uint32_t timeout = HAL_GetTick();

    while(1)
    {
        // 3. Add a tiny delay so the SPI bus isn't hammered thousands of times per millisecond
        HAL_Delay(1);

        // 4. Read the status register safely
        status = NRF24_ReadRegister(STATUS);

        if(status & TX_DS)
        {
            NRF24_WriteRegister(STATUS, TX_DS); // Clear Data Sent flag
            break;
        }

        if(status & MAX_RT)
        {
            NRF24_WriteRegister(STATUS, MAX_RT); // Clear Max Retry flag
            NRF24_FlushTX();
            break;
        }

        // 5. Hard timeout fallback (100ms max)
        if((HAL_GetTick() - timeout) > 100)
        {
            NRF24_FlushTX();
            break;
        }
    }

    NRF_CE_LOW(); // Final safety check
}

/*-----------------------------------------------------------
    Data Ready
-----------------------------------------------------------*/
uint8_t NRF24_DataReady(void)
{
    return (NRF24_ReadRegister(STATUS) & RX_DR);
}

/*-----------------------------------------------------------
    Read Payload
-----------------------------------------------------------*/
void NRF24_ReadPayload(uint8_t *data,uint8_t length)
{
    uint8_t cmd = R_RX_PAYLOAD;

    NRF_CSN_LOW();

    HAL_SPI_Transmit(&hspi1,&cmd,1,HAL_MAX_DELAY);
    HAL_SPI_Receive(&hspi1,data,length,HAL_MAX_DELAY);

    NRF_CSN_HIGH();

    NRF24_WriteRegister(STATUS,RX_DR);
}
