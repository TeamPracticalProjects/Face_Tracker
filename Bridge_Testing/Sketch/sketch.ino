// SPDX-FileCopyrightText: Copyright (C) ARDUINO SRL (http://www.arduino.cc)
//
// SPDX-License-Identifier: MPL-2.0

#include <Arduino_RouterBridge.h>

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);

    Bridge.begin();
    // Serial.begin(9600);
    Monitor.begin();
    Bridge.provide("move_servos", move_servos);

    // Monitor.println("\nMonitor is started .... \n");
}

void loop() {
}

void move_servos(String coord_str) {
    // print the received string to the Monitor
    // Monitor.print("Received string: ");
    //  Monitor.println(coord_str);

    // Serial.println(coord_str);
    Monitor.println(coord_str);
    digitalWrite(LED_BUILTIN, LOW);
    delay(400);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(400);
    digitalWrite(LED_BUILTIN, LOW);
    delay(400);
    digitalWrite(LED_BUILTIN, HIGH);

  
}