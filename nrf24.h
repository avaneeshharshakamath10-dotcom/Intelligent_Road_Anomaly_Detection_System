/*
 * nrf24.h
 *
 * Created on: Jul 10, 2026
 * Author: avane
 */

#ifndef INC_NRF24_H_
#define INC_NRF24_H_

#include "main.h"
#include "spi.h"
#include "gpio.h"

/*==========================================================
 * nRF24L01 COMMANDS
 *==========================================================*/
#define R_REGISTER          0x00
#define W_REGISTER          0x20
#define REGISTER_MASK       0x1F

#define R_RX_PAYLOAD        0x61
#define W_TX_PAYLOAD        0xA0

#define FLUSH_TX            0xE1
#define FLUSH_RX            0xE2
#define REUSE_TX_PL         0xE3

#define NOP                 0xFF

/*==========================================================
 * nRF24L01 REGISTERS
 *==========================================================*/
#define CONFIG              0x00
#define EN_AA               0x01
#define EN_RXADDR           0x02
#define SETUP_AW            0x03
#define SETUP_RETR          0x04
#define RF_CH               0x05
#define RF_SETUP            0x06
#define STATUS              0x07
#define OBSERVE_TX          0x08
#define RPD                 0x09

#define RX_ADDR_P0          0x0A
#define RX_ADDR_P1          0x0B

#define TX_ADDR             0x10

#define RX_PW_P0            0x11

#define FIFO_STATUS         0x17

/*==========================================================
 * CONFIG REGISTER BITS
 *==========================================================*/
#define MASK_RX_DR          (1 << 6)
#define MASK_TX_DS          (1 << 5)
#define MASK_MAX_RT         (1 << 4)

#define EN_CRC              (1 << 3)
#define CRCO                (1 << 2)
#define PWR_UP              (1 << 1)
#define PRIM_RX             (1 << 0)

/*==========================================================
 * STATUS REGISTER BITS
 *==========================================================*/
#define RX_DR               (1 << 6)
#define TX_DS               (1 << 5)
#define MAX_RT              (1 << 4)

/*==========================================================
 * CE CONTROL
 *==========================================================*/
#define NRF_CE_HIGH() HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET)
#define NRF_CE_LOW()  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET)

/*==========================================================
 * CSN CONTROL
 *==========================================================*/
#define NRF_CSN_HIGH() HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_SET)
#define NRF_CSN_LOW()  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_1, GPIO_PIN_RESET)

/*==========================================================
 * SPI HANDLE
 *==========================================================*/
extern SPI_HandleTypeDef hspi1;

/*==========================================================
 * DRIVER FUNCTIONS
 *==========================================================*/
uint8_t NRF24_ReadRegister(uint8_t reg);

void NRF24_WriteRegister(uint8_t reg, uint8_t value);

void NRF24_WriteBuffer(uint8_t reg,
                       uint8_t *data,
                       uint8_t length);

void NRF24_Init(void);

uint8_t NRF24_GetStatus(void);

void NRF24_Send(uint8_t *data,
                uint8_t length);

void NRF24_FlushTX(void);
void NRF24_FlushRX(void);
void NRF24_ReadPayload(uint8_t *data, uint8_t length);
uint8_t NRF24_DataReady(void);

#endif /* INC_NRF24_H_ */
