#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/spi_master.h"
#include "esp_event.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "nvs.h"
#include "nvs_flash.h"

// 小策略网络（4->64->64->2 MLP）：forward_policy(obs, action)
// 由 rl/export_policy_header.py 从训练好的权重导出：
//     python rl/export_policy_header.py rl/weights/eei_pure_rl_v1.zip -o firmware/main/policy_network.h
#include "policy_network.h"

// Wi-Fi 凭据放在 wifi_config.h 里（不进版本库）；首次使用从示例文件复制：
//     cp main/wifi_config.example.h main/wifi_config.h
// 没有该文件时自动退回示例文件，此时是占位符 SSID，连不上会进入配网模式。
#if defined(__has_include)
#  if __has_include("wifi_config.h")
#    include "wifi_config.h"
#  else
#    include "wifi_config.example.h"
#  endif
#else
#  include "wifi_config.example.h"
#endif

#define WIFI_SSID EEI_WIFI_SSID
#define WIFI_PASSWORD EEI_WIFI_PASSWORD
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_MAX_RETRY 10
#define WIFI_SETUP_SSID "BMI160-Setup"
#define WIFI_SETUP_PASSWORD "12345678"
#define WIFI_SCAN_MAX_AP 20

#define SPI_HOST SPI2_HOST
#define SPI_SCLK_GPIO GPIO_NUM_4
#define SPI_MOSI_GPIO GPIO_NUM_5
#define SPI_MISO_GPIO GPIO_NUM_7
#define SPI_CS_GPIO GPIO_NUM_6
#define SPI_CLOCK_HZ 1000000
#define SENSOR_PERIOD_MS 5
#define WS_PUSH_DIVIDER 10
#define FILTER_ALPHA 0.995f

#define SERVO_GPIO GPIO_NUM_8
#define SERVO2_GPIO GPIO_NUM_9
#define SERVO_FREQUENCY_HZ 50
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500
#define SERVO_PERIOD_US 20000

#define BMI160_REG_CHIP_ID 0x00
#define BMI160_CHIP_ID 0xD1
#define BMI160_REG_GYR_X_L 0x0C
#define BMI160_REG_ACC_CONF 0x40
#define BMI160_REG_ACC_RANGE 0x41
#define BMI160_REG_GYR_CONF 0x42
#define BMI160_REG_GYR_RANGE 0x43
#define BMI160_REG_CMD 0x7E
#define BMI160_CMD_ACC_NORMAL 0x11
#define BMI160_CMD_GYR_NORMAL 0x15
#define BMI160_ACC_RANGE_2G 0x03
#define BMI160_GYR_RANGE_250DPS 0x03
#define BMI160_ODR_200HZ 0x29

typedef enum {
    ATTITUDE_FUSION,
    ATTITUDE_GYRO,
    ATTITUDE_ACCEL,
    ATTITUDE_ACCEL_GYRO_YAW,
} attitude_mode_t;

static const char *TAG = "imu_web";
static EventGroupHandle_t wifi_events;
static spi_device_handle_t bmi160;
static httpd_handle_t http_server;
static int wifi_retries;
static char wifi_ssid[32] = WIFI_SSID;
static char wifi_password[64] = WIFI_PASSWORD;
static bool provisioning_mode;
static bool initial_scan_started;
static wifi_ap_record_t wifi_scan_records[WIFI_SCAN_MAX_AP];
static volatile float current_roll;
static volatile float current_pitch;
static volatile float current_yaw;
static volatile float current_gx;
static volatile float current_gy;
static volatile float current_gz;
// Bridge addition: raw accelerometer (g) in the same (post-invert) frame as gx/gy/gz.
// Written by the bridge to compute the IMU mounting rotation on the PC side.
static volatile float current_ax;
static volatile float current_ay;
static volatile float current_az;
// ★ IMU 适配：把传感器坐标系的读数换算到「机身坐标系」，使输出与上游模型训练时的约定一致。
//   body_R = 水平校准 × 安装矩阵（bridge/config.json 里标定得到；直立姿态下机体 z 轴朝上）。
//   直立时换算结果为 (0,0,+1) → roll=pitch=0，和作者那台机器一样。
static const float body_R[3][3] = {
    {-0.00081f, -0.98566f, 0.16877f},
    {-0.99995f, -0.00081f, -0.00958f},
    {0.00958f, -0.16877f, -0.98561f},
};
static volatile float body_roll_deg;   // 机身坐标系 roll（deg，作者模型的观测）
static volatile float body_pitch_deg;  // 机身坐标系 pitch（deg）

// ★ 板上起立策略状态（逻辑完全照作者的 Arduino/robo03/robo03.ino）
#define GETUP_PERIOD_US 20000        // 50 Hz，与训练一致
#define GETUP_TIMEOUT_US 30000000    // 单次最长 30 s（原 14 s）
#define GETUP_DONE_TILT_RAD 0.25f    // 判定"站好了"
#define GETUP_DONE_HOLD_US 2000000   // 需持续 2 s（原 700 ms）
#define AUTO_FALL_TILT_RAD 0.75f     // 判定"跌倒了"
#define AUTO_FALL_HOLD_US 300000     // 需持续 300 ms
#define AUTO_RESTART_COOLDOWN_US 1500000
#define TARGET_DELTA_RAD 0.08f
#define TARGET_LIMIT_RAD 1.55f

static volatile bool getup_active;

static void servo_set_angle(ledc_channel_t channel, volatile int *state, int angle);  // 前置声明
// 这两个变量在文件后面才定义（带初值 90），这里先做暂定声明以便策略任务提前使用
static volatile int servo_angle;
static volatile int servo2_angle;
static volatile bool auto_getup = true;      // 跌倒自动起立（方便你反复推倒测试）
static volatile float policy_target1, policy_target2;   // 目标关节角（rad）
static int64_t last_policy_us;
static int64_t getup_start_us;
static int64_t upright_since_us;
static int64_t fall_since_us;
static int64_t last_getup_end_us;

static float clampf_local(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void getup_start(void)
{
    policy_target1 = 0.0f;
    policy_target2 = 0.0f;
    getup_active = true;
    getup_start_us = esp_timer_get_time();
    upright_since_us = 0;
    last_policy_us = 0;
    ESP_LOGI(TAG, "GETUP start");
}

static void getup_stop(void)
{
    getup_active = false;
    last_getup_end_us = esp_timer_get_time();
    fall_since_us = 0;
    ESP_LOGI(TAG, "GETUP stop");
}

// ★ 起立策略任务体：由传感器任务每 5 ms 调用一次，内部按 50 Hz 出动作
static void getup_tick(void)
{
    int64_t now_us = esp_timer_get_time();
    float roll = body_roll_deg * 0.0174532925f;
    float pitch = body_pitch_deg * 0.0174532925f;
    bool fallen = (fabsf(roll) > AUTO_FALL_TILT_RAD) || (fabsf(pitch) > AUTO_FALL_TILT_RAD);

    if (auto_getup && !getup_active) {
        if (now_us - last_getup_end_us < AUTO_RESTART_COOLDOWN_US) {
            fall_since_us = 0;
        } else if (fallen) {
            if (fall_since_us == 0) fall_since_us = now_us;
            if (now_us - fall_since_us > AUTO_FALL_HOLD_US) getup_start();
        } else {
            fall_since_us = 0;
        }
    }
    if (!getup_active) return;
    if (last_policy_us != 0 && now_us - last_policy_us < GETUP_PERIOD_US) return;
    last_policy_us = now_us;

    float obs[OBS_DIM] = {roll, pitch, policy_target1, policy_target2};
    float action[ACTION_DIM];
    forward_policy(obs, action);

    policy_target1 = clampf_local(policy_target1 + clampf_local(action[0], -1.0f, 1.0f) * TARGET_DELTA_RAD,
                                  -TARGET_LIMIT_RAD, TARGET_LIMIT_RAD);
    policy_target2 = clampf_local(policy_target2 + clampf_local(action[1], -1.0f, 1.0f) * TARGET_DELTA_RAD,
                                  -TARGET_LIMIT_RAD, TARGET_LIMIT_RAD);
    // 你的标定：cmd = 90 - deg(joint)（center=90, direction=-1, scale=1）
    servo_set_angle(LEDC_CHANNEL_0, &servo_angle, (int)lroundf(90.0f - policy_target1 * 57.29578f));
    servo_set_angle(LEDC_CHANNEL_1, &servo2_angle, (int)lroundf(90.0f - policy_target2 * 57.29578f));

    if (fabsf(roll) < GETUP_DONE_TILT_RAD && fabsf(pitch) < GETUP_DONE_TILT_RAD) {
        if (upright_since_us == 0) upright_since_us = now_us;
        if (now_us - upright_since_us > GETUP_DONE_HOLD_US) getup_stop();
    } else {
        upright_since_us = 0;
    }
    if (now_us - getup_start_us > GETUP_TIMEOUT_US) {
        ESP_LOGW(TAG, "getup timeout, stop");
        getup_stop();
    }
}

static esp_err_t getup_post(httpd_req_t *req)
{
    char query[32];
    char value[8];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "value", value, sizeof(value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing value");
        return ESP_FAIL;
    }
    if (strcmp(value, "1") == 0) getup_start();
    else if (strcmp(value, "0") == 0) { auto_getup = false; getup_stop(); }
    else if (strcmp(value, "auto") == 0) auto_getup = true;
    else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "value must be 1/0/auto");
        return ESP_FAIL;
    }
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}
static volatile bool reset_requested;
static volatile attitude_mode_t attitude_mode = ATTITUDE_ACCEL_GYRO_YAW;
static volatile bool invert_gx;
static volatile bool invert_gy;
static volatile bool invert_gz;
static volatile int ws_fd = -1;
static volatile int servo_angle = 90;
static volatile int servo2_angle = 90;

static void start_webserver(void);
static void start_provisioning_ap(void);
static void apply_sta_config(void);

static const char *attitude_mode_name(attitude_mode_t mode)
{
    switch (mode) {
        case ATTITUDE_GYRO: return "gyro";
        case ATTITUDE_ACCEL: return "accel";
        case ATTITUDE_ACCEL_GYRO_YAW: return "accel_yaw";
        default: return "fusion";
    }
}

static float wrap_degrees(float angle)
{
    while (angle > 180.0f) angle -= 360.0f;
    while (angle < -180.0f) angle += 360.0f;
    return angle;
}

static float blend_angle(float predicted, float measured)
{
    return wrap_degrees(predicted + (1.0f - FILTER_ALPHA) *
                        wrap_degrees(measured - predicted));
}

static void servo_set_angle(ledc_channel_t channel, volatile int *state, int angle)
{
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;
    uint32_t pulse_us = SERVO_MIN_US + (uint32_t)(SERVO_MAX_US - SERVO_MIN_US) * angle / 180;
    uint32_t duty = pulse_us * ((1U << LEDC_TIMER_14_BIT) - 1) / SERVO_PERIOD_US;
    ESP_ERROR_CHECK(ledc_set_duty(LEDC_LOW_SPEED_MODE, channel, duty));
    ESP_ERROR_CHECK(ledc_update_duty(LEDC_LOW_SPEED_MODE, channel));
    *state = angle;
}

static void servo_init(void)
{
    const ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE, .duty_resolution = LEDC_TIMER_14_BIT,
        .timer_num = LEDC_TIMER_0, .freq_hz = SERVO_FREQUENCY_HZ, .clk_cfg = LEDC_AUTO_CLK,
    };
    const ledc_channel_config_t channel = {
        .gpio_num = SERVO_GPIO, .speed_mode = LEDC_LOW_SPEED_MODE, .channel = LEDC_CHANNEL_0,
        .intr_type = LEDC_INTR_DISABLE, .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0,
    };
    const ledc_channel_config_t channel2 = {
        .gpio_num = SERVO2_GPIO, .speed_mode = LEDC_LOW_SPEED_MODE, .channel = LEDC_CHANNEL_1,
        .intr_type = LEDC_INTR_DISABLE, .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));
    ESP_ERROR_CHECK(ledc_channel_config(&channel));
    ESP_ERROR_CHECK(ledc_channel_config(&channel2));
    servo_set_angle(LEDC_CHANNEL_0, &servo_angle, servo_angle);
    servo_set_angle(LEDC_CHANNEL_1, &servo2_angle, servo2_angle);
    ESP_LOGI(TAG, "Servo PWM: GPIO%d=%d deg, GPIO%d=%d deg, %d Hz", SERVO_GPIO, servo_angle,
             SERVO2_GPIO, servo2_angle, SERVO_FREQUENCY_HZ);
}

static const char INDEX_HTML[] =
    "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
    "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>BMI160 姿态</title><style>"
    ":root{font-family:Inter,system-ui,-apple-system,\"Segoe UI\",sans-serif;color:#edf1f2;background:#171a1d}"
    "*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#171a1d;overflow:hidden}"
    "main{height:100vh;display:grid;grid-template-columns:minmax(0,1fr) 270px}#scene{position:relative;min-width:0}canvas{display:block;width:100%;height:100%}"
    "aside{border-left:1px solid #3c4548;background:#202528;padding:26px 22px;display:flex;flex-direction:column;gap:26px}"
    "header{display:flex;align-items:center;justify-content:space-between;gap:12px}h1{margin:0;font-size:1.05rem;font-weight:650;letter-spacing:0}.tools{display:flex;align-items:center;gap:10px}"
    "#status{display:flex;align-items:center;gap:7px;font-size:.74rem;color:#aeb9bb}.dot{width:8px;height:8px;border-radius:50%;background:#757e80}.dot.live{background:#41d38b;box-shadow:0 0 14px #41d38b}"
    ".readout{border-top:1px solid #3c4548;padding-top:16px}.label{display:block;color:#8f9a9c;font-size:.72rem;margin-bottom:7px}.value{font-size:1.65rem;line-height:1;font-variant-numeric:tabular-nums;font-weight:600}.value small{font-size:.72rem;color:#8f9a9c;font-weight:500;margin-left:4px}"
    ".axes{margin-top:auto;border-top:1px solid #3c4548;padding-top:18px;display:grid;gap:11px;font-variant-numeric:tabular-nums}.axis{display:flex;justify-content:space-between;font-size:.8rem}.axis b{font-weight:600}.x{color:#f06e68}.y{color:#58c8a1}.z{color:#5da6e9}"
    "#reset{border:1px solid #4e5c60;border-radius:4px;padding:6px 9px;background:#2d3538;color:#dce4e5;font:inherit;font-size:.72rem;cursor:pointer}#reset:hover{border-color:#41d38b;color:#41d38b}"
    ".modes,.inverts{display:grid;grid-template-columns:repeat(2,1fr);gap:5px}.mode,.invert{border:1px solid #4e5c60;border-radius:4px;padding:6px 3px;background:#2d3538;color:#aeb9bb;font:inherit;font-size:.68rem;cursor:pointer}.mode.active,.invert.active{border-color:#41d38b;color:#41d38b}"
    ".inverts{grid-template-columns:repeat(3,1fr)}"
    ".servo{border-top:1px solid #3c4548;padding-top:16px;display:grid;gap:9px}.servo-head{display:flex;justify-content:space-between;color:#aeb9bb;font-size:.75rem}.servo strong{color:#edf1f2;font-size:1rem}.servo input{width:100%;accent-color:#41d38b}"
    "@media(max-width:700px){main{grid-template-columns:1fr;grid-template-rows:minmax(0,1fr) 260px}aside{border-left:0;border-top:1px solid #3c4548;padding:18px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.axes{margin:0;border-top:0;padding:0;grid-column:1/-1;grid-template-columns:repeat(3,1fr)}.readout{padding-top:0;border-top:0}.value{font-size:1.25rem}.servo{grid-column:1/-1;padding-top:10px;gap:5px}}"
    "</style></head><body><main><section id=\"scene\"></section><aside><header><h1>BMI160 姿态</h1><div class=\"tools\"><span id=\"status\"><i class=\"dot\"></i>等待数据</span><button id=\"reset\" type=\"button\">复位</button></div></header>"
    "<div class=\"modes\" role=\"group\" aria-label=\"姿态模式\"><button class=\"mode\" data-mode=\"fusion\" type=\"button\">融合</button><button class=\"mode\" data-mode=\"gyro\" type=\"button\">仅陀螺仪</button><button class=\"mode\" data-mode=\"accel\" type=\"button\">仅加速度计</button><button class=\"mode active\" data-mode=\"accel_yaw\" type=\"button\">倾角 + Yaw</button></div>"
    "<div class=\"inverts\" role=\"group\" aria-label=\"陀螺仪方向\"><button class=\"invert\" data-axis=\"gx\" type=\"button\">GX 反向</button><button class=\"invert\" data-axis=\"gy\" type=\"button\">GY 反向</button><button class=\"invert\" data-axis=\"gz\" type=\"button\">GZ 反向</button></div>"
    "<div class=\"servo\"><div class=\"servo-head\"><span>GPIO8 舵机</span><strong id=\"servoAngle\">90deg</strong></div><input id=\"servo\" type=\"range\" min=\"0\" max=\"180\" value=\"90\" aria-label=\"舵机角度\"></div>"
    "<div class=\"servo\"><div class=\"servo-head\"><span>GPIO9 舵机</span><strong id=\"servo2Angle\">90deg</strong></div><input id=\"servo2\" type=\"range\" min=\"0\" max=\"180\" value=\"90\" aria-label=\"GPIO9 舵机角度\"></div>"
    "<div class=\"readout\"><span class=\"label\">横滚 Roll</span><span class=\"value\" id=\"roll\">0.0<small>deg</small></span></div>"
    "<div class=\"readout\"><span class=\"label\">俯仰 Pitch</span><span class=\"value\" id=\"pitch\">0.0<small>deg</small></span></div>"
    "<div class=\"readout\"><span class=\"label\">偏航 Yaw</span><span class=\"value\" id=\"yaw\">0.0<small>deg</small></span></div>"
    "<div class=\"axes\"><span class=\"axis x\"><b>GX</b><i id=\"gx\">0.0</i></span><span class=\"axis y\"><b>GY</b><i id=\"gy\">0.0</i></span><span class=\"axis z\"><b>GZ</b><i id=\"gz\">0.0</i></span></div>"
    "</aside></main><script type=\"importmap\">{\"imports\":{\"three\":\"https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js\"}}</script><script type=\"module\">"
    "import * as THREE from 'three';"
    "import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/controls/OrbitControls.js';"
    "const mount=document.querySelector('#scene'),renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));mount.append(renderer.domElement);"
    "const scene=new THREE.Scene();scene.background=new THREE.Color(0x171a1d);const camera=new THREE.PerspectiveCamera(38,1,.1,100);camera.position.set(6,5,7);"
    "const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.target.set(0,0,0);controls.minDistance=4;controls.maxDistance=13;"
    "scene.add(new THREE.HemisphereLight(0xeaf4f2,0x22272a,2.4));const light=new THREE.DirectionalLight(0xffffff,2.4);light.position.set(4,7,4);scene.add(light);"
    "const floor=new THREE.GridHelper(16,16,0x354044,0x293033);floor.position.y=-1.05;scene.add(floor);const rig=new THREE.Group();scene.add(rig);"
    "function box(w,h,d,color,x,y,z){const m=new THREE.MeshStandardMaterial({color,roughness:.55,metalness:.08});const o=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),m);o.position.set(x,y,z);rig.add(o);return o}"
    "box(5,.35,3.4,0xe5eaeb,0,0,0);box(4.86,.08,3.25,0xcbd3d5,0,.22,0);"
    "const holeMat=new THREE.MeshStandardMaterial({color:0x6a7478,roughness:.9});for(let x=-2.1;x<=2.1;x+=.28)for(let z=-1.25;z<=1.25;z+=.28){if(Math.abs(z)<.27)continue;const h=new THREE.Mesh(new THREE.CylinderGeometry(.035,.035,.025,10),holeMat);h.rotation.x=Math.PI/2;h.position.set(x,.275,z);rig.add(h)}"
    "box(1.6,.28,2.65,0x1c2528,-.55,.47,0);box(1.12,.13,1.45,0x9da6a8,-.55,.68,0);"
    "const chip=box(.95,.11,.78,0x638b8d,-.55,.81,0);const imu=box(.74,.09,.98,0x8c3a78,1.42,.48,.65);box(.22,.05,.22,0x242a2c,1.42,.56,.65);"
    "const pinMat=new THREE.MeshStandardMaterial({color:0xe0b653,metalness:.65,roughness:.28});for(let i=0;i<8;i++){const p=new THREE.Mesh(new THREE.BoxGeometry(.05,.05,.14),pinMat);p.position.set(1.12+i*.085,.54,.65-.58);rig.add(p)}"
    "const axis=new THREE.AxesHelper(1);axis.position.set(1.42,.61,.65);rig.add(axis);"
    "function resize(){const w=mount.clientWidth,h=mount.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}addEventListener('resize',resize);resize();"
    "const el=n=>document.querySelector('#'+n),status=el('status');let target={r:0,p:0,y:0};"
    "const servo=el('servo'),servoAngle=el('servoAngle'),servo2=el('servo2'),servo2Angle=el('servo2Angle');let servoDragging=false,servo2Dragging=false;function fmt(v){return (v>=0?'+':'')+v.toFixed(1)}function update(d){target={r:d.r,p:d.p,y:d.y};el('roll').innerHTML=fmt(d.r)+'<small>deg</small>';el('pitch').innerHTML=fmt(d.p)+'<small>deg</small>';el('yaw').innerHTML=fmt(d.y)+'<small>deg</small>';el('gx').textContent=fmt(d.gx);el('gy').textContent=fmt(d.gy);el('gz').textContent=fmt(d.gz);if(!servoDragging){servo.value=d.s;servoAngle.textContent=d.s+'deg'}if(!servo2Dragging){servo2.value=d.s2;servo2Angle.textContent=d.s2+'deg'}document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active',x.dataset.mode===d.m));document.querySelector('[data-axis=\"gx\"]').classList.toggle('active',d.ix);document.querySelector('[data-axis=\"gy\"]').classList.toggle('active',d.iy);document.querySelector('[data-axis=\"gz\"]').classList.toggle('active',d.iz);status.innerHTML='<i class=\"dot live\"></i>实时连接'}"
    "let ws_live=false,poll_busy=false,ws_socket=null,control_live=false,control_socket=null;async function poll(){if(ws_live||poll_busy)return;poll_busy=true;try{update(await fetch('/imu',{cache:'no-store'}).then(r=>r.json()))}catch(e){status.innerHTML='<i class=\"dot\"></i>正在重连'}finally{poll_busy=false;setTimeout(poll,100)}}"
    "function connectWs(){const ws=new WebSocket('ws://'+location.host+'/ws');ws_socket=ws;ws.onopen=()=>{ws_live=true};ws.onmessage=e=>{try{update(JSON.parse(e.data))}catch(_) {}};ws.onerror=()=>ws.close();ws.onclose=()=>{if(ws_socket===ws)ws_socket=null;ws_live=false;setTimeout(connectWs,500);poll()}}function connectControlWs(){const ws=new WebSocket('ws://'+location.host+'/control');control_socket=ws;ws.onopen=()=>{control_live=true};ws.onerror=()=>ws.close();ws.onclose=()=>{if(control_socket===ws)control_socket=null;control_live=false;setTimeout(connectControlWs,500)}}connectWs();connectControlWs();setTimeout(poll,1000);function sendServoCommand(channel,value){if(control_live&&control_socket&&control_socket.readyState===WebSocket.OPEN&&control_socket.bufferedAmount<4096){control_socket.send('servo'+channel+':'+value);return Promise.resolve()}return fetch('/servo'+(channel===2?'2':'')+'?angle='+value,{method:'POST',cache:'no-store'})}"
    "el('reset').onclick=async()=>{try{await fetch('/reset',{method:'POST',cache:'no-store'});target={r:0,p:0,y:0};status.innerHTML='<i class=\"dot live\"></i>已设为正向'}catch(e){status.innerHTML='<i class=\"dot\"></i>复位失败'}};"
    "document.querySelectorAll('.mode').forEach(b=>b.onclick=async()=>{try{await fetch('/mode?value='+b.dataset.mode,{method:'POST',cache:'no-store'});document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active',x===b));status.innerHTML='<i class=\"dot live\"></i>已切换模式'}catch(e){status.innerHTML='<i class=\"dot\"></i>切换失败'}});"
    "document.querySelectorAll('.invert').forEach(b=>b.onclick=async()=>{const on=!b.classList.contains('active');try{await fetch('/invert?axis='+b.dataset.axis+'&value='+(on?1:0),{method:'POST',cache:'no-store'});b.classList.toggle('active',on)}catch(e){status.innerHTML='<i class=\"dot\"></i>方向切换失败'}});"
    "let servo1Busy=false,servo1Queued=null,servo1Scheduled=false,servo1Last=0,servo2Busy=false,servo2Queued=null,servo2Scheduled=false,servo2Last=0;function scheduleServo1(force=false){const value=servo.value;servoAngle.textContent=value+'deg';servo1Queued=value;if(servo1Busy||servo1Scheduled)return;servo1Scheduled=true;const delay=force?0:Math.max(0,30-(performance.now()-servo1Last));setTimeout(flushServo1,delay)}async function flushServo1(){servo1Scheduled=false;if(servo1Busy||servo1Queued===null)return;const value=servo1Queued;servo1Queued=null;servo1Busy=true;try{await sendServoCommand(1,value)}catch(e){status.innerHTML='<i class=\"dot\"></i>舵机请求失败'}finally{servo1Last=performance.now();servo1Busy=false;if(servo1Queued!==null)scheduleServo1(false)}}function scheduleServo2(force=false){const value=servo2.value;servo2Angle.textContent=value+'deg';servo2Queued=value;if(servo2Busy||servo2Scheduled)return;servo2Scheduled=true;const delay=force?0:Math.max(0,30-(performance.now()-servo2Last));setTimeout(flushServo2,delay)}async function flushServo2(){servo2Scheduled=false;if(servo2Busy||servo2Queued===null)return;const value=servo2Queued;servo2Queued=null;servo2Busy=true;try{await sendServoCommand(2,value)}catch(e){status.innerHTML='<i class=\"dot\"></i>舵机2请求失败'}finally{servo2Last=performance.now();servo2Busy=false;if(servo2Queued!==null)scheduleServo2(false)}}servo.addEventListener('pointerdown',()=>servoDragging=true);servo.addEventListener('pointerup',()=>{servoDragging=false;scheduleServo1(true)});servo.addEventListener('input',()=>scheduleServo1(false));servo2.addEventListener('pointerdown',()=>servo2Dragging=true);servo2.addEventListener('pointerup',()=>{servo2Dragging=false;scheduleServo2(true)});servo2.addEventListener('input',()=>scheduleServo2(false));"
    "function loop(){requestAnimationFrame(loop);const q=new THREE.Quaternion().setFromEuler(new THREE.Euler(target.r*Math.PI/180,target.y*Math.PI/180,-target.p*Math.PI/180,'YXZ'));rig.quaternion.slerp(q,.45);controls.update();renderer.render(scene,camera)}loop();"
    "</script></body></html>";

static const char WIFI_SETUP_HTML[] =
    "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>BMI160 配网</title><style>body{max-width:420px;margin:48px auto;padding:0 20px;font-family:system-ui,sans-serif;color:#17211f}h1{margin-bottom:6px}p{color:#53615d;line-height:1.55}label{display:block;margin-top:18px;font-weight:600}input{width:100%;box-sizing:border-box;margin-top:7px;padding:12px;border:1px solid #abb8b3;border-radius:6px;font:inherit}button{width:100%;margin-top:24px;padding:12px;border:0;border-radius:6px;background:#16895b;color:#fff;font:inherit;font-weight:600}</style></head><body>"
    "<h1>BMI160 配网</h1><p>请输入要连接的 Wi-Fi。保存后设备会关闭配网热点并尝试连接。</p>"
    "<form method=\"post\" action=\"/wifi\"><label>Wi-Fi 名称（SSID）<input name=\"ssid\" maxlength=\"31\" required autofocus></label>"
    "<label>Wi-Fi 密码<input name=\"password\" type=\"password\" maxlength=\"63\" placeholder=\"开放网络可留空\"></label><button type=\"submit\">保存并连接</button></form></body></html>";

static esp_err_t bmi160_write(uint8_t reg, uint8_t value)
{
    const uint8_t tx[] = {(uint8_t)(reg & 0x7F), value};
    spi_transaction_t transaction = {.length = sizeof(tx) * 8, .tx_buffer = tx};
    return spi_device_transmit(bmi160, &transaction);
}

static esp_err_t bmi160_read(uint8_t reg, uint8_t *data, size_t length)
{
    uint8_t tx[13] = {0};
    uint8_t rx[13] = {0};
    if (length > sizeof(tx) - 1) return ESP_ERR_INVALID_SIZE;
    tx[0] = reg | 0x80;
    spi_transaction_t transaction = {.length = (length + 1) * 8, .tx_buffer = tx, .rx_buffer = rx};
    esp_err_t err = spi_device_transmit(bmi160, &transaction);
    if (err == ESP_OK) memcpy(data, &rx[1], length);
    return err;
}

static int16_t read_le_i16(const uint8_t *data)
{
    return (int16_t)((uint16_t)data[0] | ((uint16_t)data[1] << 8));
}

static void bmi160_init(void)
{
    const spi_bus_config_t bus_config = {
        .mosi_io_num = SPI_MOSI_GPIO, .miso_io_num = SPI_MISO_GPIO, .sclk_io_num = SPI_SCLK_GPIO,
        .quadwp_io_num = -1, .quadhd_io_num = -1, .max_transfer_sz = 16,
    };
    const spi_device_interface_config_t device_config = {
        .clock_speed_hz = SPI_CLOCK_HZ, .mode = 0, .spics_io_num = SPI_CS_GPIO, .queue_size = 1,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI_HOST, &bus_config, SPI_DMA_CH_AUTO));
    ESP_ERROR_CHECK(spi_bus_add_device(SPI_HOST, &device_config, &bmi160));
    uint8_t discard;
    ESP_ERROR_CHECK(bmi160_read(0x7F, &discard, 1));
    vTaskDelay(pdMS_TO_TICKS(2));
    uint8_t chip_id = 0;
    ESP_ERROR_CHECK(bmi160_read(BMI160_REG_CHIP_ID, &chip_id, 1));
    ESP_ERROR_CHECK(chip_id == BMI160_CHIP_ID ? ESP_OK : ESP_ERR_INVALID_RESPONSE);
    ESP_LOGI(TAG, "BMI160 SPI CHIP_ID=0x%02X", chip_id);
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_CMD, BMI160_CMD_ACC_NORMAL));
    vTaskDelay(pdMS_TO_TICKS(10));
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_CMD, BMI160_CMD_GYR_NORMAL));
    vTaskDelay(pdMS_TO_TICKS(100));
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_ACC_CONF, BMI160_ODR_200HZ));
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_GYR_CONF, BMI160_ODR_200HZ));
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_ACC_RANGE, BMI160_ACC_RANGE_2G));
    ESP_ERROR_CHECK(bmi160_write(BMI160_REG_GYR_RANGE, BMI160_GYR_RANGE_250DPS));
    ESP_LOGI(TAG, "BMI160 output data rate: 200 Hz");
}

static esp_err_t root_get(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html; charset=utf-8");
    if (provisioning_mode) return httpd_resp_send(req, WIFI_SETUP_HTML, HTTPD_RESP_USE_STRLEN);
    return httpd_resp_send(req, INDEX_HTML, HTTPD_RESP_USE_STRLEN);
}

static bool url_decode(char *output, size_t output_size, const char *input)
{
    size_t out = 0;
    for (size_t in = 0; input[in] != '\0'; ++in) {
        char value = input[in];
        if (value == '+') value = ' ';
        else if (value == '%' && input[in + 1] && input[in + 2]) {
            char hex[3] = {input[in + 1], input[in + 2], '\0'};
            char *end = NULL;
            long decoded = strtol(hex, &end, 16);
            if (*end != '\0') return false;
            value = (char)decoded;
            in += 2;
        }
        if (out + 1 >= output_size) return false;
        output[out++] = value;
    }
    output[out] = '\0';
    return true;
}

static esp_err_t wifi_post(httpd_req_t *req)
{
    if (!provisioning_mode || req->content_len <= 0 || req->content_len >= 180) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid Wi-Fi setup request");
        return ESP_FAIL;
    }
    char form[180];
    int received = 0;
    while (received < req->content_len) {
        int ret = httpd_req_recv(req, form + received, req->content_len - received);
        if (ret <= 0) return ESP_FAIL;
        received += ret;
    }
    form[received] = '\0';
    char encoded_ssid[100] = {0};
    char encoded_password[140] = {0};
    if (httpd_query_key_value(form, "ssid", encoded_ssid, sizeof(encoded_ssid)) != ESP_OK ||
        httpd_query_key_value(form, "password", encoded_password, sizeof(encoded_password)) != ESP_OK ||
        !url_decode(wifi_ssid, sizeof(wifi_ssid), encoded_ssid) ||
        !url_decode(wifi_password, sizeof(wifi_password), encoded_password) || wifi_ssid[0] == '\0') {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid SSID or password");
        return ESP_FAIL;
    }

    nvs_handle_t handle;
    esp_err_t err = nvs_open("wifi_cfg", NVS_READWRITE, &handle);
    if (err == ESP_OK) {
        err = nvs_set_str(handle, "ssid", wifi_ssid);
        if (err == ESP_OK) err = nvs_set_str(handle, "password", wifi_password);
        if (err == ESP_OK) err = nvs_commit(handle);
        nvs_close(handle);
    }
    if (err != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Could not save Wi-Fi settings");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "text/html; charset=utf-8");
    httpd_resp_sendstr(req, "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><p>Wi-Fi 已保存，设备正在连接。请回到你的路由器网络，并查看串口日志中的新 IP 地址。</p>");
    ESP_LOGI(TAG, "Wi-Fi settings saved for SSID: %s", wifi_ssid);
    provisioning_mode = false;
    wifi_retries = 0;
    apply_sta_config();
    esp_wifi_disconnect();
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    esp_wifi_connect();
    return ESP_OK;
}

static esp_err_t imu_get(httpd_req_t *req)
{
    char payload[384];
    int length = snprintf(payload, sizeof(payload), "{\"r\":%.2f,\"p\":%.2f,\"y\":%.2f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"m\":\"%s\",\"ix\":%d,\"iy\":%d,\"iz\":%d,\"s\":%d,\"s2\":%d,\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"roll\":%.2f,\"pitch\":%.2f,\"t1\":%.4f,\"t2\":%.4f}",
                          current_roll, current_pitch, current_yaw, current_gx, current_gy, current_gz,
                          attitude_mode_name(attitude_mode), invert_gx, invert_gy, invert_gz, servo_angle, servo2_angle,
                          current_ax, current_ay, current_az,
                          body_roll_deg, body_pitch_deg,
                          (90.0f - (float)servo_angle) * 0.0174532925f,
                          (90.0f - (float)servo2_angle) * 0.0174532925f);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Cache-Control", "no-store");
    return httpd_resp_send(req, payload, length);
}

static esp_err_t reset_post(httpd_req_t *req)
{
    reset_requested = true;
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static esp_err_t ws_post_handshake(httpd_req_t *req)
{
    ws_fd = httpd_req_to_sockfd(req);
    ESP_LOGI(TAG, "WebSocket client connected fd=%d", ws_fd);
    return ESP_OK;
}

static esp_err_t ws_handler(httpd_req_t *req)
{
    httpd_ws_frame_t frame = {0};
    frame.type = HTTPD_WS_TYPE_TEXT;
    esp_err_t err = httpd_ws_recv_frame(req, &frame, 0);
    if (err == ESP_OK && frame.type == HTTPD_WS_TYPE_CLOSE) ws_fd = -1;
    return err;
}

static esp_err_t control_ws_handler(httpd_req_t *req)
{
    httpd_ws_frame_t frame = {.type = HTTPD_WS_TYPE_TEXT};
    esp_err_t err = httpd_ws_recv_frame(req, &frame, 0);
    if (err != ESP_OK || frame.type == HTTPD_WS_TYPE_CLOSE || frame.type != HTTPD_WS_TYPE_TEXT ||
        frame.len == 0 || frame.len >= 64) {
        return err;
    }

    uint8_t payload[64];
    frame.payload = payload;
    err = httpd_ws_recv_frame(req, &frame, frame.len);
    if (err != ESP_OK) return err;
    payload[frame.len] = '\0';

    int channel = 0;
    int angle = -1;
    if (sscanf((char *)payload, "servo%d:%d", &channel, &angle) != 2 || angle < 0 || angle > 180 ||
        (channel != 1 && channel != 2)) {
        return ESP_OK;
    }

    if (channel == 1) servo_set_angle(LEDC_CHANNEL_0, &servo_angle, angle);
    else servo_set_angle(LEDC_CHANNEL_1, &servo2_angle, angle);

    char reply[16];
    int reply_len = snprintf(reply, sizeof(reply), "ok%d:%d", channel, angle);
    httpd_ws_frame_t response = {.type = HTTPD_WS_TYPE_TEXT, .payload = (uint8_t *)reply, .len = reply_len};
    return httpd_ws_send_frame(req, &response);
}

static esp_err_t control_ws_post_handshake(httpd_req_t *req)
{
    return ESP_OK;
}

static esp_err_t mode_post(httpd_req_t *req)
{
    char query[32];
    char value[12];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "value", value, sizeof(value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing mode");
        return ESP_FAIL;
    }
    if (strcmp(value, "fusion") == 0) attitude_mode = ATTITUDE_FUSION;
    else if (strcmp(value, "gyro") == 0) attitude_mode = ATTITUDE_GYRO;
    else if (strcmp(value, "accel") == 0) attitude_mode = ATTITUDE_ACCEL;
    else if (strcmp(value, "accel_yaw") == 0) attitude_mode = ATTITUDE_ACCEL_GYRO_YAW;
    else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid mode");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Attitude mode: %s", attitude_mode_name(attitude_mode));
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static esp_err_t invert_post(httpd_req_t *req)
{
    char query[48];
    char axis[8];
    char value[8];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "axis", axis, sizeof(axis)) != ESP_OK ||
        httpd_query_key_value(query, "value", value, sizeof(value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing axis or value");
        return ESP_FAIL;
    }
    bool enabled = strcmp(value, "1") == 0;
    if (strcmp(axis, "gx") == 0) invert_gx = enabled;
    else if (strcmp(axis, "gy") == 0) invert_gy = enabled;
    else if (strcmp(axis, "gz") == 0) invert_gz = enabled;
    else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid axis");
        return ESP_FAIL;
    }
    reset_requested = true;
    ESP_LOGI(TAG, "Gyro inversion %s=%d", axis, enabled);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static esp_err_t servo_post(httpd_req_t *req)
{
    char query[32];
    char value[8];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "angle", value, sizeof(value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing angle");
        return ESP_FAIL;
    }
    char *end = NULL;
    long angle = strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || angle < 0 || angle > 180) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Angle must be 0-180");
        return ESP_FAIL;
    }
    servo_set_angle(LEDC_CHANNEL_0, &servo_angle, (int)angle);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static esp_err_t servo2_post(httpd_req_t *req)
{
    char query[32];
    char value[8];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "angle", value, sizeof(value)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Missing angle");
        return ESP_FAIL;
    }
    char *end = NULL;
    long angle = strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || angle < 0 || angle > 180) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Angle must be 0-180");
        return ESP_FAIL;
    }
    servo_set_angle(LEDC_CHANNEL_1, &servo2_angle, (int)angle);
    httpd_resp_set_type(req, "application/json");
    return httpd_resp_sendstr(req, "{\"ok\":true}");
}

static void start_webserver(void)
{
    if (http_server != NULL) return;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.lru_purge_enable = true;
    config.max_uri_handlers = 12;
    if (httpd_start(&http_server, &config) != ESP_OK) return;
    const httpd_uri_t root = {.uri = "/", .method = HTTP_GET, .handler = root_get};
    const httpd_uri_t wifi = {.uri = "/wifi", .method = HTTP_POST, .handler = wifi_post};
    const httpd_uri_t imu = {.uri = "/imu", .method = HTTP_GET, .handler = imu_get};
    const httpd_uri_t reset = {.uri = "/reset", .method = HTTP_POST, .handler = reset_post};
    const httpd_uri_t mode = {.uri = "/mode", .method = HTTP_POST, .handler = mode_post};
    const httpd_uri_t invert = {.uri = "/invert", .method = HTTP_POST, .handler = invert_post};
    const httpd_uri_t servo = {.uri = "/servo", .method = HTTP_POST, .handler = servo_post};
    const httpd_uri_t servo2 = {.uri = "/servo2", .method = HTTP_POST, .handler = servo2_post};
    const httpd_uri_t getup = {.uri = "/getup", .method = HTTP_POST, .handler = getup_post};
    const httpd_uri_t ws = {.uri = "/ws", .method = HTTP_GET, .handler = ws_handler, .is_websocket = true,
                            .ws_post_handshake_cb = ws_post_handshake};
    const httpd_uri_t control = {.uri = "/control", .method = HTTP_GET, .handler = control_ws_handler, .is_websocket = true,
                                 .ws_post_handshake_cb = control_ws_post_handshake};
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &root));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &wifi));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &imu));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &reset));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &mode));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &invert));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &servo));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &servo2));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &getup));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &ws));
    ESP_ERROR_CHECK(httpd_register_uri_handler(http_server, &control));
}

static void load_wifi_credentials(void)
{
    nvs_handle_t handle;
    size_t ssid_size = sizeof(wifi_ssid);
    size_t password_size = sizeof(wifi_password);
    if (nvs_open("wifi_cfg", NVS_READONLY, &handle) != ESP_OK) return;
    esp_err_t ssid_err = nvs_get_str(handle, "ssid", wifi_ssid, &ssid_size);
    esp_err_t password_err = nvs_get_str(handle, "password", wifi_password, &password_size);
    nvs_close(handle);
    if (ssid_err != ESP_OK || password_err != ESP_OK || wifi_ssid[0] == '\0') {
        snprintf(wifi_ssid, sizeof(wifi_ssid), "%s", WIFI_SSID);
        snprintf(wifi_password, sizeof(wifi_password), "%s", WIFI_PASSWORD);
    } else {
        ESP_LOGI(TAG, "Loaded saved Wi-Fi settings for SSID: %s", wifi_ssid);
    }
}

static void apply_sta_config(void)
{
    wifi_config_t config = {0};
    memcpy(config.sta.ssid, wifi_ssid, sizeof(config.sta.ssid));
    memcpy(config.sta.password, wifi_password, sizeof(config.sta.password));
    config.sta.threshold.authmode = wifi_password[0] ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &config));
}

static void start_provisioning_ap(void)
{
    if (provisioning_mode) return;
    provisioning_mode = true;
    wifi_config_t ap_config = {0};
    snprintf((char *)ap_config.ap.ssid, sizeof(ap_config.ap.ssid), "%s", WIFI_SETUP_SSID);
    snprintf((char *)ap_config.ap.password, sizeof(ap_config.ap.password), "%s", WIFI_SETUP_PASSWORD);
    ap_config.ap.ssid_len = strlen(WIFI_SETUP_SSID);
    ap_config.ap.channel = 1;
    ap_config.ap.max_connection = 4;
    ap_config.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    start_webserver();
    ESP_LOGW(TAG, "Wi-Fi '%s' was not found. Connect to %s (password: %s), then open http://192.168.4.1/",
             wifi_ssid, WIFI_SETUP_SSID, WIFI_SETUP_PASSWORD);
}

static void start_wifi_scan(void)
{
    wifi_scan_config_t scan_config = {0};
    initial_scan_started = true;
    if (esp_wifi_scan_start(&scan_config, false) != ESP_OK) {
        ESP_LOGW(TAG, "Wi-Fi scan could not start; opening setup AP");
        start_provisioning_ap();
    }
}

static void wifi_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        if (!initial_scan_started) start_wifi_scan();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_SCAN_DONE) {
        uint16_t count = 0;
        ESP_ERROR_CHECK(esp_wifi_scan_get_ap_num(&count));
        uint16_t record_count = count > WIFI_SCAN_MAX_AP ? WIFI_SCAN_MAX_AP : count;
        ESP_ERROR_CHECK(esp_wifi_scan_get_ap_records(&record_count, wifi_scan_records));
        bool found = false;
        for (uint16_t i = 0; i < record_count; ++i) {
            if (strcmp((char *)wifi_scan_records[i].ssid, wifi_ssid) == 0) {
                found = true;
                break;
            }
        }
        if (found) {
            wifi_retries = 0;
            ESP_LOGI(TAG, "Wi-Fi '%s' found; connecting", wifi_ssid);
            esp_wifi_connect();
        } else {
            start_provisioning_ap();
        }
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (!provisioning_mode && wifi_retries++ < WIFI_MAX_RETRY) esp_wifi_connect();
        else if (!provisioning_mode) start_provisioning_ap();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = event_data;
        char ip[16];
        esp_ip4addr_ntoa(&event->ip_info.ip, ip, sizeof(ip));
        ESP_LOGI(TAG, "Open http://%s/ on the PC", ip);
        xEventGroupSetBits(wifi_events, WIFI_CONNECTED_BIT);
        start_webserver();
    }
}

static void wifi_init(void)
{
    wifi_events = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_handler, NULL, NULL));
    load_wifi_credentials();
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    apply_sta_config();
    ESP_ERROR_CHECK(esp_wifi_start());
}

static void sensor_task(void *arg)
{
    uint8_t raw[12];
    float roll = 0, pitch = 0, yaw = 0;
    float roll_zero = 0, pitch_zero = 0, yaw_zero = 0;
    float gyro_roll = 0, gyro_pitch = 0, gyro_yaw = 0;
    float gyro_roll_zero = 0, gyro_pitch_zero = 0, gyro_yaw_zero = 0;
    float acc_roll_zero = 0, acc_pitch_zero = 0;
    int64_t last_us = esp_timer_get_time();
    uint32_t samples = 0;
    uint32_t ws_ticks = 0;
    while (true) {
        int64_t now_us = esp_timer_get_time();
        float dt = (now_us - last_us) / 1000000.0f;
        last_us = now_us;
        if (dt <= 0 || dt > 0.1f) dt = SENSOR_PERIOD_MS / 1000.0f;
        if (bmi160_read(BMI160_REG_GYR_X_L, raw, sizeof(raw)) == ESP_OK) {
            float gx = read_le_i16(&raw[0]) / 131.2f;
            float gy = read_le_i16(&raw[2]) / 131.2f;
            float gz = read_le_i16(&raw[4]) / 131.2f;
            if (invert_gx) gx = -gx;
            if (invert_gy) gy = -gy;
            if (invert_gz) gz = -gz;
            float ax = read_le_i16(&raw[6]) / 16384.0f;
            float ay = read_le_i16(&raw[8]) / 16384.0f;
            float az = read_le_i16(&raw[10]) / 16384.0f;
            // Keep accelerometer and gyro axes in the same user-selected frame.
            if (invert_gx) ax = -ax;
            if (invert_gy) ay = -ay;
            if (invert_gz) az = -az;
            // ★ IMU 适配：换算到机身坐标系（矩阵见文件上方 body_R 的注释）
            float bx = body_R[0][0] * ax + body_R[0][1] * ay + body_R[0][2] * az;
            float by = body_R[1][0] * ax + body_R[1][1] * ay + body_R[1][2] * az;
            float bz = body_R[2][0] * ax + body_R[2][1] * ay + body_R[2][2] * az;
            float bnorm = sqrtf(bx * bx + by * by + bz * bz);
            if (bnorm > 0.0001f) {
                bx /= bnorm; by /= bnorm; bz /= bnorm;
            }
            // 与上游 quat_to_roll_pitch 的约定一致：roll = atan2(uy, uz)，pitch = -atan2(ux, hypot)
            body_roll_deg = atan2f(by, bz) * 57.29578f;
            body_pitch_deg = -atan2f(bx, sqrtf(by * by + bz * bz)) * 57.29578f;
            getup_tick();   // ★ 板上起立策略（50 Hz 出动作，逻辑同作者的 robo03.ino）
            // Use -Z as the reference for both axes so Roll and Pitch each use
            // the full -180 to +180 degree display range.
            float roll_acc = atan2f(-ay, -az) * 57.29578f;
            float pitch_acc = atan2f(ax, -az) * 57.29578f;
            roll = blend_angle(wrap_degrees(roll + gx * dt), roll_acc);
            // pitch_acc and the BMI160 Y-axis gyro use the same positive direction.
            pitch = blend_angle(wrap_degrees(pitch + gy * dt), pitch_acc);
            yaw += gz * dt;
            gyro_roll += gx * dt;
            gyro_pitch += gy * dt;
            gyro_yaw += gz * dt;
            if (yaw > 180) yaw -= 360;
            if (yaw < -180) yaw += 360;
            if (reset_requested) {
                roll_zero = roll;
                pitch_zero = pitch;
                yaw_zero = yaw;
                gyro_roll_zero = gyro_roll;
                gyro_pitch_zero = gyro_pitch;
                gyro_yaw_zero = gyro_yaw;
                acc_roll_zero = roll_acc;
                acc_pitch_zero = pitch_acc;
                reset_requested = false;
                ESP_LOGI(TAG, "Orientation zero point updated");
            }
            if (attitude_mode == ATTITUDE_GYRO) {
                current_roll = wrap_degrees(gyro_roll - gyro_roll_zero);
                current_pitch = wrap_degrees(gyro_pitch - gyro_pitch_zero);
                current_yaw = wrap_degrees(gyro_yaw - gyro_yaw_zero);
            } else if (attitude_mode == ATTITUDE_ACCEL) {
                current_roll = wrap_degrees(roll_acc - acc_roll_zero);
                current_pitch = wrap_degrees(pitch_acc - acc_pitch_zero);
                current_yaw = 0;
            } else if (attitude_mode == ATTITUDE_ACCEL_GYRO_YAW) {
                current_roll = wrap_degrees(roll_acc - acc_roll_zero);
                current_pitch = wrap_degrees(pitch_acc - acc_pitch_zero);
                current_yaw = wrap_degrees(gyro_yaw - gyro_yaw_zero);
            } else {
                current_roll = wrap_degrees(roll - roll_zero);
                current_pitch = wrap_degrees(pitch - pitch_zero);
                current_yaw = wrap_degrees(yaw - yaw_zero);
            }
            current_gx = gx;
            current_gy = gy;
            current_gz = gz;
            current_ax = ax;
            current_ay = ay;
            current_az = az;
            if (ws_fd >= 0 && http_server != NULL && (++ws_ticks % WS_PUSH_DIVIDER == 0)) {
                char payload[384];
                int length = snprintf(payload, sizeof(payload), "{\"r\":%.2f,\"p\":%.2f,\"y\":%.2f,\"gx\":%.2f,\"gy\":%.2f,\"gz\":%.2f,\"m\":\"%s\",\"ix\":%d,\"iy\":%d,\"iz\":%d,\"s\":%d,\"s2\":%d,\"ax\":%.4f,\"ay\":%.4f,\"az\":%.4f,\"roll\":%.2f,\"pitch\":%.2f,\"t1\":%.4f,\"t2\":%.4f}",
                                      current_roll, current_pitch, current_yaw, current_gx, current_gy, current_gz,
                                      attitude_mode_name(attitude_mode), invert_gx, invert_gy, invert_gz, servo_angle, servo2_angle,
                                      current_ax, current_ay, current_az,
                                      body_roll_deg, body_pitch_deg,
                                      (90.0f - (float)servo_angle) * 0.0174532925f,
                                      (90.0f - (float)servo2_angle) * 0.0174532925f);
                httpd_ws_frame_t frame = {.type = HTTPD_WS_TYPE_TEXT, .payload = (uint8_t *)payload, .len = length};
                if (httpd_ws_send_frame_async(http_server, ws_fd, &frame) != ESP_OK) ws_fd = -1;
            }
            if (++samples % 50 == 0) ESP_LOGI(TAG, "IMU streaming");
        } else if (samples++ % 50 == 0) {
            ESP_LOGW(TAG, "BMI160 SPI data read failed");
        }
        vTaskDelay(pdMS_TO_TICKS(SENSOR_PERIOD_MS));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "BMI160 SPI: SCLK=GPIO4 MOSI=GPIO5 MISO=GPIO7 CS=GPIO6");
    servo_init();
    bmi160_init();
    esp_err_t nvs_err = nvs_flash_init();
    if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES || nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(nvs_err);
    wifi_init();
    xTaskCreate(sensor_task, "sensor", 4096, NULL, 5, NULL);
}
