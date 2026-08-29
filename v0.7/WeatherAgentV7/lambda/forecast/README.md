# forecast Lambda (v0.7 gateway target)

Fronts Open-Meteo's forecast endpoint. It exists because the AgentCore Gateway's
HTTP client can't consume that host's `Transfer-Encoding: chunked` responses —
a Lambda target sidesteps the issue entirely.

Create it in your own account, then point the gateway target at the ARN it prints:

```bash
zip -j function.zip index.py

aws iam create-role --role-name WeatherAgentV7ForecastLambdaRole \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name WeatherAgentV7ForecastLambdaRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws lambda create-function --function-name WeatherAgentV7Forecast \
  --runtime python3.12 --handler index.handler --timeout 15 \
  --role arn:aws:iam::<YOUR_ACCOUNT_ID>:role/WeatherAgentV7ForecastLambdaRole \
  --zip-file fileb://function.zip
```

Then register it as a gateway target:

```bash
agentcore add gateway-target --type lambda-function-arn --name ForecastTool \
  --lambda-arn arn:aws:lambda:us-east-1:<YOUR_ACCOUNT_ID>:function:WeatherAgentV7Forecast \
  --tool-schema-file specs/forecast-lambda-tools.json --gateway WeatherGateway
```
