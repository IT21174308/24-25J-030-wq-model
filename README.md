# 24-25J-030-wq-model  
Flow Customization model to predict last stage values.

## Build Docker Image  
```bash
docker build -t flow-customization-model .
```
## Run Docker Container
```bash
docker run -d -p 5000:5000 --name flow-container flow-customization-model
```


## API Endpoint (Test via Postman)
* URL: (POST) http://localhost:5000/predict
* JSON Body:
```bash
{
  "raw_turbidity": 6,
  "raw_ph": 1,
  "raw_conductivity": 2
}
```
