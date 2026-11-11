/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "spi.h"
#include "usart.h"
#include "gpio.h"
#include "nrf24.h"
#include <stdio.h>
#include <string.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct
{
    uint8_t systemToken;
    uint8_t vehicleID;
    uint8_t alertCode;
    uint8_t frameCounter;
    uint8_t checksum;
    uint8_t reserved;
} VANET_Packet;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* USER CODE BEGIN PV */
uint8_t rxByte;
char rxBuffer[50];
volatile uint8_t rxIndex = 0;
VANET_Packet txPacket;
volatile uint8_t frameCounter = 0;

// Flag to tell the main loop there is a command ready to process
volatile uint8_t commandReady = 0;
volatile uint8_t commandType = 0; // 1 = POTHOLE, 2 = HUMP
/* USER CODE END PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
uint8_t CalculateChecksum(VANET_Packet *pkt)
{
    return pkt->systemToken ^
           pkt->vehicleID ^
           pkt->alertCode ^
           pkt->frameCounter ^
           pkt->reserved;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_SPI1_Init();
  HAL_UART_Transmit(&huart2,
                    (uint8_t*)"STM32 Started\r\n",
                    15,
                    HAL_MAX_DELAY);
  /* USER CODE BEGIN 2 */
  NRF24_Init();
  char msg[250];

  sprintf(msg, "STATUS DIRECT = 0x%02X\r\n", NRF24_GetStatus());

  HAL_UART_Transmit(&huart2,
                    (uint8_t*)msg,
                    strlen(msg),
                    HAL_MAX_DELAY);


  sprintf(msg,
  "CONFIG     = 0x%02X\r\n"
  "EN_AA      = 0x%02X\r\n"
  "EN_RXADDR  = 0x%02X\r\n"
  "SETUP_AW   = 0x%02X\r\n"
  "SETUP_RETR = 0x%02X\r\n"
  "RF_CH      = 0x%02X\r\n"
  "RF_SETUP   = 0x%02X\r\n"
  "STATUS     = 0x%02X\r\n\r\n",

  NRF24_ReadRegister(CONFIG),
  NRF24_ReadRegister(EN_AA),
  NRF24_ReadRegister(EN_RXADDR),
  NRF24_ReadRegister(SETUP_AW),
  NRF24_ReadRegister(SETUP_RETR),
  NRF24_ReadRegister(RF_CH),
  NRF24_ReadRegister(RF_SETUP),
  NRF24_ReadRegister(STATUS));

  HAL_UART_Transmit(&huart2,
                    (uint8_t*)msg,
                    strlen(msg),
                    HAL_MAX_DELAY);

  uint8_t txAddr[5] = {'N','O','D','E','1'};

  NRF24_WriteBuffer(TX_ADDR, txAddr, 5);
  NRF24_WriteBuffer(RX_ADDR_P0, txAddr, 5);

  NRF24_WriteRegister(0x11, sizeof(VANET_Packet)); // RX_PW_P0
  NRF24_WriteRegister(0x07, 0x70);                 // Clear interrupt flags
  HAL_Delay(100);

  sprintf(msg,
          "\r\nCONFIG=0x%02X STATUS=0x%02X RF_CH=%d\r\n",
          NRF24_ReadRegister(CONFIG),
          NRF24_ReadRegister(STATUS),
          NRF24_ReadRegister(RF_CH));

  HAL_UART_Transmit(&huart2,
                    (uint8_t*)msg,
                    strlen(msg),
                    HAL_MAX_DELAY);

  HAL_UART_Transmit(&huart2,
                    (uint8_t*)"Waiting for command...\r\n",
                    strlen("Waiting for command...\r\n"),
                    HAL_MAX_DELAY);

  HAL_UART_Receive_IT(&huart2, &rxByte, 1);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  /* Infinite loop */
    /* USER CODE BEGIN WHILE */
  /* Infinite loop */
    /* USER CODE BEGIN WHILE */
    while (1)
    {
        if (commandReady)
        {
            // 1. Construct the packet based on what the interrupt caught
            txPacket.systemToken = 0xAA;
            txPacket.vehicleID = 0x01;
            txPacket.alertCode = commandType;
            txPacket.frameCounter = frameCounter++;
            txPacket.reserved = 0x00;
            txPacket.checksum = CalculateChecksum(&txPacket);

            // 2. Print debug log
            char debug[100];
            sprintf(debug, "Sending %s! ID=%d Frame=%d Check=0x%02X\r\n",
                    (commandType == 1) ? "POTHOLE" : "HUMP",
                    txPacket.vehicleID, txPacket.frameCounter, txPacket.checksum);
            HAL_UART_Transmit(&huart2, (uint8_t*)debug, strlen(debug), HAL_MAX_DELAY);

            // 3. Send out over nRF24 safely in main execution context.
            //    With Auto-Ack disabled (required for broadcast to multiple receivers),
            //    there's no hardware retransmission on a missed packet -- so we send the
            //    same payload a few times in quick succession instead. Losing 1 of 3 tries
            //    is far less noticeable than losing a single one-shot transmission.
            for (uint8_t resend = 0; resend < 3; resend++)
            {
                NRF24_Send((uint8_t*)&txPacket, sizeof(VANET_Packet));
                HAL_Delay(5);
            }

            // 4. Safely clear buffer flags by disabling interrupts for a split microsecond
            __disable_irq();
            memset(rxBuffer, 0, sizeof(rxBuffer));
            rxIndex = 0;
            commandReady = 0;
            commandType = 0;
            __enable_irq();
        }
      /* USER CODE END WHILE */

      /* USER CODE BEGIN 3 */
    }
    /* USER CODE END 3 */
    /* USER CODE END 3 */
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();
  __HAL_RCC_LSEDRIVE_CONFIG(RCC_LSEDRIVE_LOW);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_LSE|RCC_OSCILLATORTYPE_MSI;
  RCC_OscInitStruct.LSEState = RCC_LSE_ON;
  RCC_OscInitStruct.MSIState = RCC_MSI_ON;
  RCC_OscInitStruct.MSICalibrationValue = 0;
  RCC_OscInitStruct.MSIClockRange = RCC_MSIRANGE_6;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_MSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 16;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Enable MSI Auto calibration
  */
  HAL_RCCEx_EnableMSIPLLMode();
}

/* USER CODE BEGIN 4 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        // Handle line endings
        if (rxByte == '\r' || rxByte == '\n')
        {
            if (rxIndex > 0)
            {
                rxBuffer[rxIndex] = '\0';

                // Just set the flags! Do NOT call NRF24_Send here!
                if(strcmp(rxBuffer, "POTHOLE") == 0)
                {
                    commandType = 1;
                    commandReady = 1;
                }
                else if(strcmp(rxBuffer, "HUMP") == 0)
                {
                    commandType = 2;
                    commandReady = 1;
                }
                else
                {
                    // Clean up unknown commands instantly
                    memset(rxBuffer, 0, sizeof(rxBuffer));
                    rxIndex = 0;
                }
            }
        }
        else
        {
            if (rxIndex < sizeof(rxBuffer) - 1)
            {
                rxBuffer[rxIndex++] = rxByte;
            }
        }

        // Keep the interrupt listener alive
        HAL_UART_Receive_IT(&huart2, &rxByte, 1);
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
