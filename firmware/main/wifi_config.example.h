// Wi-Fi 配置示例。
//
// 用法：复制成 wifi_config.h 并填上自己的网络，
//     cp main/wifi_config.example.h main/wifi_config.h
// wifi_config.h 已在 .gitignore 里，不会被提交。
//
// 连不上这个网络时，固件会自动开启配网热点（见 WIFI_SETUP_SSID / WIFI_SETUP_PASSWORD），
// 用手机连上后打开 http://192.168.4.1/ 填入新的 Wi-Fi 即可，凭据保存在 NVS 里。

#pragma once

#define EEI_WIFI_SSID "your-wifi-ssid"
#define EEI_WIFI_PASSWORD "your-wifi-password"
