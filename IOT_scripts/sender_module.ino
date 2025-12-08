#include <WiFi.h>
#include <esp_now.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>

// -------------------- PIN DEFINITIONS --------------------
#define DHTPIN 12
#define DHTTYPE DHT11
#define MQ2_PIN 34
#define FLAME_PIN 27

// -------------------- SENSOR THRESHOLDS --------------------
const float Vref = 3.3;
const int ADC_MAX = 4095;
const float RL = 20000.0;       // 20kΩ load resistor (per MQ2 datasheet)
const float R0 = 10000.0;       // Calibrated clean-air resistance (adjust experimentally)
const float MQ2_THRESHOLD_RATIO = 0.7;  // Rs/R0 ≤ 0.7 indicates smoke/gas
const int FLAME_TRIGGER_STATE = LOW;    // LOW = flame detected
const float TEMP_CRITICAL = 45.0;       // °C temperature threshold for fire

// -------------------- DATA STRUCTURE --------------------
typedef struct struct_message {
  float temperature;
  float humidity;
  float gas_ratio;
  bool flame_detected;
  bool fire_alert;
} struct_message;

struct_message dataToSend;

// -------------------- OBJECTS --------------------
DHT dht(DHTPIN, DHTTYPE);

// 🔹 Receiver MAC Address (replace with your receiver’s)
uint8_t receiverMAC[] = {0x78, 0x21, 0x84, 0xBB, 0x2C, 0x0C};

// -------------------- ESP-NOW SEND CALLBACK --------------------
// ✅ Updated callback for ESP-IDF v5.x (Arduino core v3+)
void onDataSent(const esp_now_send_info_t *info, esp_now_send_status_t status) {
  Serial.print("ESP-NOW Send Status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

// -------------------- UTILITY FUNCTIONS --------------------
float adcToVoltage(int adc) {
  return ((float)adc / ADC_MAX) * Vref;
}

float voltageToRs(float Vout) {
  if (Vout < 0.001) return 1e9;
  return RL * ((Vref - Vout) / Vout);
}

// -------------------- SETUP --------------------
void setup() {
  Serial.begin(115200);
  dht.begin();

  pinMode(MQ2_PIN, INPUT);
  pinMode(FLAME_PIN, INPUT);

  // Wi-Fi must be in STA mode for ESP-NOW
  WiFi.mode(WIFI_STA);
  Serial.println("Initializing ESP-NOW...");

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW!");
    while (true);
  }

  // Register the callback
  esp_now_register_send_cb(onDataSent);

  // Add peer (receiver)
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer!");
    while (true);
  }

  Serial.println("ESP-NOW initialized successfully!");
}

// -------------------- MAIN LOOP --------------------
void loop() {
  // --- Read DHT11 ---
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  // --- Read MQ2 ---
  int adc = analogRead(MQ2_PIN);
  float Vout = adcToVoltage(adc);
  float Rs = voltageToRs(Vout);
  float ratio = Rs / R0;

  // --- Read Flame Sensor ---
  bool flameDetected = (digitalRead(FLAME_PIN) == FLAME_TRIGGER_STATE);

  // --- Fire / Gas Alert Logic ---
  bool gasAlert = (ratio <= MQ2_THRESHOLD_RATIO);
  bool tempAlert = (!isnan(temp) && temp >= TEMP_CRITICAL);
  bool fireAlert = (gasAlert || flameDetected || tempAlert);

  // --- Fill Structure ---
  dataToSend.temperature = temp;
  dataToSend.humidity = hum;
  dataToSend.gas_ratio = ratio;
  dataToSend.flame_detected = flameDetected;
  dataToSend.fire_alert = fireAlert;

  // --- Debug Output ---
  Serial.println("------ SENSOR DATA ------");
  Serial.printf("Temperature: %.1f °C\n", temp);
  Serial.printf("Humidity: %.1f %%\n", hum);
  Serial.printf("Gas Rs/R0: %.2f\n", ratio);
  Serial.printf("Flame Detected: %s\n", flameDetected ? "YES" : "NO");
  Serial.printf("Fire Alert: %s\n", fireAlert ? "🔥 YES" : "NO");
  Serial.println("--------------------------");

  // --- Send via ESP-NOW ---
  esp_err_t result = esp_now_send(receiverMAC, (uint8_t *) &dataToSend, sizeof(dataToSend));
  if (result == ESP_OK) Serial.println("Data sent successfully!\n");
  else Serial.println("Error sending data.\n");

  delay(5000);  // send every 5 seconds
}
