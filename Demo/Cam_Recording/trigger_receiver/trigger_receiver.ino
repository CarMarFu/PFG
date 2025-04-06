void setup() {
  pinMode(13,OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if(Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'H') {
      digitalWrite(13, HIGH);
    } else if (command == 'L') {
      digitalWrite(13, LOW);
    }
  }
}
