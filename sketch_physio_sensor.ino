const int GSR=A0;
const int ECG=A1;
const float V_SUPPLY = 5.0;      // Supply voltage to the divider (e.g. Arduino 5V)
const float R_REF = 200000.0;    // Reference resistor or potentiometer setting in ohms
const float ADC_DENOM = 1024.0;  // For 10-bit ADC (Arduino Uno etc.)
const unsigned int sampleCount = 10; // Packet of 10 measurements
const unsigned long sampleInterval = 5000;  // microseconds between samples
int adcValue=0;
float gSkin_uS_average;
int val2Send=0;
int val2SendGSR=0;
int heartValue=0;
int rec=0;
int gsrOn=0; // Real GSR mode
int ecgOn=0; // Real ECG mode
int synthOn=0; // Synthetic ECG mode - will send data from dataArray1[]
int synthGSROn=0; // Synthetic GSR mode - will send data from dataArray2[]
//float sum_gSkin_uS;
int dataArray1[] = {326, 340, 336, 323, 333, 346, 342, 332, 344, 356, 349, 343, 353, 362, 352, 349, 360, 368, 353, 353, 364, 368, 353, 352, 362, 363, 342, 343, 353, 347, 329, 332, 341, 333, 319, 329, 336, 325, 318, 327, 336, 321, 316, 328, 340, 323, 316, 329, 337, 322, 323, 334, 336, 318, 321, 335, 337, 321, 325, 340, 337, 324, 330, 339, 330, 320, 331, 340, 327, 321, 330, 341, 326, 320, 333, 342, 327, 321, 335, 341, 325, 327, 338, 338, 322, 328, 342, 339, 322, 328, 343, 340, 328, 337, 345, 334, 321, 330, 338, 324, 322, 332, 341, 327, 323, 332, 340, 326, 325, 333, 333, 311, 308, 326, 357, 383, 404, 402, 353, 265, 251, 301, 335, 329, 333, 341, 328, 320, 331, 339, 326, 323, 335, 343, 326, 326, 339, 344, 327, 332, 342, 341, 324, 329, 343, 342, 330, 338, 352, 348, 338, 348, 362, 352, 345, 357, 369, 357, 355, 366, 370, 355, 351, 362, 366, 344, 341, 353, 351, 331, 331, 339, 336, 317, 323, 332, 327, 314, 320, 330, 322, 312, 319, 331, 321, 313, 325, 332, 318, 317, 329, 336, 319, 318};
int array1Size = sizeof(dataArray1) / sizeof(dataArray1[0]);
float dataArray2[] = {39.67, 39.67, 39.59, 39.44, 39.41, 39.25, 39.21, 39.11, 39.06, 39.02, 38.91, 38.76, 38.72, 38.64, 38.53, 38.38, 38.28, 38.35, 38.31, 38.13, 38.02, 37.98, 38.02, 37.95, 37.71, 37.71, 37.59, 37.45, 37.34, 37.24, 37.24, 38.06, 39.81, 41.81, 43.91, 45.69, 47.24, 48.11, 48.67, 48.89, 49.06, 49.01, 48.95, 48.89, 49.01, 48.95, 49.10, 48.83, 47.16, 43.11};
int array2Size = sizeof(dataArray2) / sizeof(dataArray2[0]);
int index = 0;  // Current index

int ecgValues[sampleCount];

void setup(){
  Serial.begin(115200);
  delay(1000);
}
 
void loop(){
  rec = Serial.read();
  switch (rec){
      // "a" sent to arduino
      case 97:
        delay(1000);
        gsrOn=1;
        rec = 0;
        break;
      // "b" sent to arduino
      case 98:
        delay(1000);
        ecgOn=1;
        rec = 0;
        break;
      // "c" sent do arduino
      case 99:
        delay(1000);
        synthOn=1;
        rec = 0;
        break;
      // "d" sent do arduino
      case 100:
        delay(1000);
        synthGSROn=1;
        rec = 0;
        break;  
      // "x" sent to arduino
      case 120:
        gsrOn=0;
        ecgOn=0;
        synthOn=0;
        synthGSROn=0;
        rec = 0;
        break;
  }     
  float sum_gSkin_uS = 0.0;
  
  for(int i=0;i<sampleCount;i++)           // Collect a packet of 10 measurements
      {
        if (gsrOn>0)
          {
            
            adcValue=analogRead(GSR);
            // --- Convert ADC to voltage ---
            float vOut = adcValue * (V_SUPPLY / ADC_DENOM);
            // --- Avoid invalid values ---
            if (vOut <= 0.0 || vOut >= (V_SUPPLY - 0.0001)) 
              {
                Serial.println("Invalid reading (out of range)");
                delayMicroseconds(sampleInterval);
                return;
              }
            
            // --- Compute skin resistance ---
            float rSkin = R_REF * (vOut / (V_SUPPLY - vOut)); 
             
            // --- Compute conductance in microSiemens ---
            float gSkin_uS = (1.0 / rSkin) * 1e6;
            sum_gSkin_uS += gSkin_uS; // 10 GSR measurements will be avaraged
          }
        
        if (ecgOn>0)
          {
            heartValue = analogRead(ECG);
            ecgValues[i] = heartValue;
          }
        if (synthOn>0)
          {
            heartValue = dataArray1[index];
            index++;
            if (index >= array1Size) {
              index = 0; 
            }
            ecgValues[i] = heartValue;
          }
        
        delayMicroseconds(sampleInterval);
      }
   if (gsrOn>0)
     {
       gSkin_uS_average = sum_gSkin_uS/10; // Avarage 10 GSR measurements
       val2SendGSR = (int16_t)(gSkin_uS_average*100); // Value converted to int, needs proper decoding
       Serial.write((byte*)&val2SendGSR, sizeof(val2SendGSR)); // Convert to binary and send out
       val2SendGSR=0;
     }

   if (synthGSROn>0)
     {
       val2SendGSR = (int16_t)(dataArray2[index]*100); // Value converted to int, needs proper decoding
       index++;  
       if (index >= array2Size) {
            index = 0;  // Loop back to start
          }
       Serial.write((byte*)&val2SendGSR, sizeof(val2SendGSR)); // Convert to binary and send out
       val2SendGSR=0;
     }

   if (ecgOn>0 || synthOn>0){
     Serial.write((byte*)ecgValues, sizeof(ecgValues)); // Convert to binary and send out
     memset(ecgValues, 0, sizeof(ecgValues));
   }
   
}
