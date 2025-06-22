void setup() {
  pinMode(8,OUTPUT);
  pinMode(9,OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if(Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'H') {
      digitalWrite(8, HIGH);
      delay(100);
      digitalWrite(8,LOW);
    } else if (command == 'L') {
      digitalWrite(9, HIGH);
      delay(100);
      digitalWrite(9,LOW);
    }
  }
}
