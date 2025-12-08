/*
  Receiver for Fire Detection System
  - Receives data via ESP-NOW from another ESP32 (sender)
  - Uploads values to Arduino IoT Cloud
  - Compatible with ESP32 core v2.0.x

  Arduino Cloud Thing: https://create.arduino.cc/cloud/things/a433a26e-1f4d-4a8a-85d7-6856ee9a9c7b

  Cloud Variables:
    float gas_ratio;
    float humidity;
    float temprature;
    bool  fire_alert;
    bool  flame_detected;
*/

#include <WiFi.h>
#include <esp_now.h>
#include "thingProperties.h"

#include "time.h"

const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 0;
const int   daylightOffset_sec = 0;


// ---------- ESP-NOW data structure ----------
#pragma pack(1)
typedef struct struct_message {
  float temperature;
  float humidity;
  float gas_ratio;
  bool  flame_detected;
  bool  fire_alert;
} struct_message;
#pragma pack()

struct_message incomingData;

// ---------- ESP-NOW callback (for ESP32 core v2.x) ----------
void onDataRecv(const uint8_t *mac, const uint8_t *incomingDataBytes, int len) {
  memcpy(&incomingData, incomingDataBytes, sizeof(incomingData));

  Serial.println("\n====== RECEIVED DATA ======");
  Serial.printf("Temperature: %.2f °C\n", incomingData.temperature);
  Serial.printf("Humidity: %.2f %%\n", incomingData.humidity);
  Serial.printf("Gas Rs/R0: %.2f\n", incomingData.gas_ratio);
  Serial.printf("Flame Detected: %s\n", incomingData.flame_detected ? "YES" : "NO");
  Serial.printf("Fire Alert: %s\n", incomingData.fire_alert ? "YES" : "NO");
  Serial.println("============================");

  // --- Update Cloud variables ---
  temprature     = incomingData.temperature;
  humidity       = incomingData.humidity;
  gas_ratio      = incomingData.gas_ratio;
  flame_detected = incomingData.flame_detected;
  fire_alert     = incomingData.fire_alert;
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(1500);

  // --- Initialize Arduino IoT Cloud ---
  initProperties();
    // Sync time manually before Cloud
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  delay(2000);
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);
  setDebugMessageLevel(2);
  ArduinoCloud.printDebugInfo();

  // --- Initialize ESP-NOW ---
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println(" ESP-NOW initialization failed!");
  } else {
    esp_now_register_recv_cb(onDataRecv);
    Serial.println(" ESP-NOW Receiver ready!");
  }
}

// ---------- Loop ----------
void loop() {
  ArduinoCloud.update();
  delay(500);
  // ESP-NOW reception handled by callback
}

// ---------- Callback stubs (required for Cloud) ----------
void onTempratureChange()   {}
void onHumidityChange()     {}
void onGasRatioChange()     {}
void onFlameDetectedChange(){}
void onFireAlertChange()    {}