idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES esp_driver_gpio esp_driver_ledc esp_driver_spi esp_event esp_http_server esp_netif esp_timer esp_wifi nvs_flash)
