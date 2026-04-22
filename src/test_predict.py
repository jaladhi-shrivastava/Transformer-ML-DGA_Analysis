from src.predict import predict_fault

sample = {
    "H2": 500,
    "CH4": 200,
    "C2H6": 100,
    "C2H4": 300,
    "C2H2": 50,
    "CO": 400,
    "CO2": 2000,
    "CH4_H2": 200/500,
    "C2H6_CH4": 100/200,
    "C2H2_C2H4": 50/300,
    "C2H4_C2H6": 300/100,
    "CO2_CO": 2000/400
}

print(predict_fault(sample))