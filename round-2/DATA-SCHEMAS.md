# NFPC Phase 2 — Data Schemas & Samples

## Table Overview
| Table | Rows | Columns | Join Key |
|-------|------|---------|----------|
| customers | 159,416 | 14 | customer_id |
| accounts | 160,153 | 22 | account_id |
| demographics | 159,416 | 9 | customer_id |
| accounts-additional | 160,153 | 2 | account_id |
| branch | 9,000 | 9 | branch_code |
| customer_account_linkage | 160,153 | 2 | customer_id ↔ account_id |
| product_details | 159,416 | 11 | customer_id |
| train_labels | 96,091 | 5 | account_id |
| test_accounts | 64,062 | 1 | account_id |
| transactions | ~400M | 8 | account_id, transaction_id |
| transactions_additional | ~400M | 9 | transaction_id |

## Relationships
```
customers ──(customer_id)──> customer_account_linkage ──(account_id)──> accounts
    │                                                                       │
    ├──> demographics                                                       ├──> transactions ──(transaction_id)──> transactions_additional
    └──> product_details                                                    ├──> train_labels / test_accounts
                                                                            ├──> accounts-additional
                                                                            └──(branch_code)──> branch
```

Note: customer_id is NOT in accounts.parquet. Must join via customer_account_linkage.

---

## customers.parquet (159,416 × 14)
| Column | dtype | Nulls | Notes |
|--------|-------|-------|-------|
| customer_id | object | 0 | CUST_NNNNNN |
| date_of_birth | object | 0 | YYYY-MM-DD |
| relationship_start_date | object | 0 | YYYY-MM-DD |
| pan_available | object | 0 | Y/N |
| aadhaar_available | object | 67,822 | Y/N, 42.5% null |
| passport_available | object | 0 | Y/N |
| mobile_banking_flag | object | 0 | Y/N |
| internet_banking_flag | object | 0 | Y/N |
| atm_card_flag | object | 0 | Y/N |
| demat_flag | object | 0 | Y/N |
| credit_card_flag | object | 0 | Y/N |
| fastag_flag | object | 0 | Y/N |
| customer_pin | int64 | 0 | Residential PIN code |
| permanent_pin | int64 | 0 | Permanent address PIN code |

Sample: CUST_000000, DOB 1950-07-15, relationship 2019-10-03, PAN=Y, Aadhaar=Y, PIN 515008

## accounts.parquet (160,153 × 22)
| Column | dtype | Nulls | Notes |
|--------|-------|-------|-------|
| account_id | object | 0 | ACCT_NNNNNN |
| account_status | object | 0 | active/frozen |
| product_code | int64 | 0 | |
| currency_code | int64 | 0 | 1=INR |
| account_opening_date | object | 0 | YYYY-MM-DD |
| branch_code | int64 | 0 | |
| branch_pin | float64 | 15,505 | |
| avg_balance | float64 | 9,468 | Can be negative (overdraft) |
| product_family | object | 0 | S/K/O |
| nomination_flag | object | 0 | Y/N |
| cheque_allowed | object | 0 | Y/N |
| cheque_availed | object | 0 | Y/N |
| num_chequebooks | int64 | 0 | |
| last_mobile_update_date | object | 135,918 | 84.9% null |
| kyc_compliant | object | 0 | Y/N |
| last_kyc_date | object | 0 | YYYY-MM-DD |
| rural_branch | object | 0 | Y/N |
| monthly_avg_balance | float64 | 9,468 | |
| quarterly_avg_balance | float64 | 0 | |
| daily_avg_balance | float64 | 0 | |
| freeze_date | object | 148,372 | 92.6% null |
| unfreeze_date | object | 156,447 | 97.7% null |

Sample: ACCT_000000, active, product_code=200, opened 2023-08-30, branch=7982, avg_bal=53720, family=K

## demographics.parquet (159,416 × 9)
| Column | dtype | Nulls | Notes |
|--------|-------|-------|-------|
| customer_id | object | 0 | |
| name | object | 0 | Full name |
| gender | object | 0 | M/F |
| address_last_update_date | object | 0 | |
| address | object | 0 | Street + city |
| phone_number | object | 0 | 91-NNNNNNNNNN |
| passbook_last_update_date | object | 47,885 | 30% null |
| joint_account_flag | object | 0 | Y/N |
| nri_flag | object | 0 | Y/N |

## transactions (~400M × 8, 396 parquet parts)
| Column | dtype | Nulls (per 1M part) | Notes |
|--------|-------|---------------------|-------|
| transaction_id | object | 0 | TXN_NNNNNNNNNN |
| account_id | object | 0 | |
| transaction_timestamp | object | 0 | ISO format |
| mcc_code | int64 | 0 | |
| channel | object | 0 | 35 channel codes |
| amount | float64 | 0 | Negative = reversal |
| txn_type | object | 0 | D/C |
| counterparty_id | object | 0 | CP_NNNNNN |

Each part has ~1M rows. 5-year window: Jul 2020 - Jun 2025.

Channel codes: UPC, UPD, END, IPM, STD, P2A, FTD, NTD, MCR, FTC, MAC, TPD, APD, CHQ, ATW, TPC, STC, OCD, RCD, IFD, ETD, NWD, CSD, IFC, PCA, MAD, CHD, RTD, CCL, OPI, CTC, SID, ASD, IAD, SCW

## transactions_additional (~400M × 9, 311 parquet parts)
| Column | dtype | Nulls (per 1.4M part) | Notes |
|--------|-------|------------------------|-------|
| transaction_id | object | 0 | Joins to transactions |
| mnemonic_code | object | 0 | Same as channel |
| latitude | float64 | 1,033,323 (73.8%) | |
| longitude | float64 | 1,033,323 (73.8%) | |
| ip_address | object | 916,078 (65.4%) | |
| balance_after_transaction | float64 | 0 | Running balance |
| part_transaction_type | object | 0 | CI/BI/IP/IC |
| atm_deposit_channel_code | object | 1,396,180 (99.7%) | CDM/CRM |
| transaction_sub_type | object | 0 | CLT_CASH/LOAN/NORMAL |

## accounts-additional.parquet (160,153 × 2)
| Column | dtype | Nulls |
|--------|-------|-------|
| account_id | object | 0 |
| scheme_code | object | 0 |

Scheme codes: PMJDY, PMSBY, PMJJBY, APY, SCSS, SSA, REGULAR

## branch.parquet (9,000 × 9)
| Column | dtype | Nulls |
|--------|-------|-------|
| branch_code | int64 | 0 |
| branch_address | object | 0 |
| branch_pin_code | int64 | 0 |
| branch_city | object | 0 |
| branch_state | object | 0 |
| branch_employee_count | int64 | 0 |
| branch_turnover | float64 | 0 |
| branch_asset_size | float64 | 0 |
| branch_type | object | 0 | urban/semi-urban/rural |

## product_details.parquet (159,416 × 11)
| Column | dtype | Nulls | Notes |
|--------|-------|-------|-------|
| customer_id | object | 0 | |
| loan_sum | float64 | 126,103 | 79.1% null (no loans) |
| loan_count | int64 | 0 | |
| cc_sum | float64 | 133,781 | 83.9% null (no CC) |
| cc_count | int64 | 0 | |
| od_sum | float64 | 0 | Can be negative |
| od_count | int64 | 0 | |
| ka_sum | float64 | 0 | |
| ka_count | int64 | 0 | |
| sa_sum | float64 | 0 | |
| sa_count | int64 | 0 | |

## train_labels.parquet (96,091 × 5)
| Column | dtype | Nulls |
|--------|-------|-------|
| account_id | object | 0 |
| is_mule | int64 | 0 | 2,683 mules, 93,408 legit |
| mule_flag_date | object | 93,408 | Only for mules |
| alert_reason | object | 93,653 | 245 mules have no reason |
| flagged_by_branch | float64 | 93,408 | Branch that flagged |

## test_accounts.parquet (64,062 × 1)
| Column | dtype | Nulls |
|--------|-------|-------|
| account_id | object | 0 |
