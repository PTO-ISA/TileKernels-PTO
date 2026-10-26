# Candidate v5 full level-0 performance

ASC/VMI ratio; three alternating-order msprof rounds. All24 forward and24 backward configs passed; this remains a historical candidate, not final acceptance.

| direction | id | dtype | hidden | clamp | weighted | ASC us | VMI us | ratio |
|---|---|---|---|---|---|---|---|---|
| forward | 0 | bf16 | 2048 | None | True | 185.870 | 184.815 | 1.0057 |
| forward | 1 | bf16 | 2048 | 0.5 | False | 263.230 | 249.013 | 1.0572 |
| forward | 2 | bf16 | 3584 | None | True | 307.459 | 304.546 | 1.0094 |
| forward | 3 | bf16 | 3584 | 0.5 | False | 416.697 | 385.308 | 1.0815 |
| forward | 4 | fp32 | 2048 | None | True | 186.743 | 185.448 | 1.0055 |
| forward | 5 | fp32 | 2048 | 0.5 | False | 261.396 | 250.546 | 1.0429 |
| forward | 6 | fp32 | 3584 | None | True | 295.041 | 295.434 | 0.9987 |
| forward | 7 | fp32 | 3584 | 0.5 | False | 406.328 | 386.594 | 1.0510 |
| forward | 8 | e4m3 | 2048 | None | True | 188.229 | 190.086 | 0.9906 |
| forward | 9 | e4m3 | 2048 | 0.5 | False | 236.488 | 272.169 | 0.8689 |
| forward | 10 | e4m3 | 3584 | None | True | 301.562 | 300.561 | 1.0033 |
| forward | 11 | e4m3 | 3584 | 0.5 | False | 343.710 | 360.890 | 0.9523 |
| forward | 12 | bf16 | 2048 | None | True | 241.353 | 240.101 | 1.0049 |
| forward | 13 | bf16 | 2048 | 0.5 | False | 336.385 | 314.447 | 1.0698 |
| forward | 14 | bf16 | 3584 | None | True | 403.246 | 401.854 | 1.0034 |
| forward | 15 | bf16 | 3584 | 0.5 | False | 535.617 | 496.265 | 1.0792 |
| forward | 16 | fp32 | 2048 | None | True | 240.599 | 243.239 | 0.9900 |
| forward | 17 | fp32 | 2048 | 0.5 | False | 328.744 | 311.167 | 1.0565 |
| forward | 18 | fp32 | 3584 | None | True | 391.921 | 391.033 | 1.0023 |
| forward | 19 | fp32 | 3584 | 0.5 | False | 521.857 | 494.930 | 1.0543 |
| forward | 20 | e4m3 | 2048 | None | True | 232.879 | 235.517 | 0.9887 |
| forward | 21 | e4m3 | 2048 | 0.5 | False | 299.903 | 345.063 | 0.8694 |
| forward | 22 | e4m3 | 3584 | None | True | 374.345 | 375.767 | 0.9962 |
| forward | 23 | e4m3 | 3584 | 0.5 | False | 441.645 | 462.494 | 0.9551 |
| backward | 0 | e4m3 | 2048 | None | True | 615.716 | 708.856 | 0.8685 |
| backward | 1 | e4m3 | 2048 | 0.5 | False | 491.104 | 546.207 | 0.8991 |
| backward | 2 | e4m3 | 3584 | None | True | 1007.959 | 1166.023 | 0.8645 |
| backward | 3 | e4m3 | 3584 | 0.5 | False | 824.452 | 930.955 | 0.8856 |
| backward | 4 | bf16 | 2048 | None | True | 444.013 | 491.066 | 0.9043 |
| backward | 5 | bf16 | 2048 | 0.5 | False | 312.723 | 348.405 | 0.8976 |
| backward | 6 | bf16 | 3584 | None | True | 731.402 | 808.218 | 0.9050 |
| backward | 7 | bf16 | 3584 | 0.5 | False | 515.387 | 595.606 | 0.8653 |
| backward | 8 | fp32 | 2048 | None | True | 516.023 | 530.469 | 0.9732 |
| backward | 9 | fp32 | 2048 | 0.5 | False | 439.859 | 443.067 | 0.9933 |
| backward | 10 | fp32 | 3584 | None | True | 817.543 | 823.705 | 0.9915 |
| backward | 11 | fp32 | 3584 | 0.5 | False | 654.139 | 659.459 | 0.9922 |
| backward | 12 | e4m3 | 2048 | None | True | 679.313 | 784.374 | 0.8660 |
| backward | 13 | e4m3 | 2048 | 0.5 | False | 542.913 | 603.417 | 0.8998 |
| backward | 14 | e4m3 | 3584 | None | True | 1119.164 | 1295.341 | 0.8640 |
| backward | 15 | e4m3 | 3584 | 0.5 | False | 883.174 | 997.082 | 0.8858 |
| backward | 16 | bf16 | 2048 | None | True | 468.032 | 517.696 | 0.9042 |
| backward | 17 | bf16 | 2048 | 0.5 | False | 345.110 | 382.791 | 0.9018 |
| backward | 18 | bf16 | 3584 | None | True | 807.723 | 892.651 | 0.9049 |
| backward | 19 | bf16 | 3584 | 0.5 | False | 552.426 | 638.133 | 0.8657 |
| backward | 20 | fp32 | 2048 | None | True | 556.825 | 572.693 | 0.9722 |
| backward | 21 | fp32 | 2048 | 0.5 | False | 475.048 | 478.310 | 0.9931 |
| backward | 22 | fp32 | 3584 | None | True | 875.469 | 882.675 | 0.9919 |
| backward | 23 | fp32 | 3584 | 0.5 | False | 723.930 | 730.733 | 0.9907 |
