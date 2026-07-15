Traceback (most recent call last):
  File "/app/Pipeline/ipd4bq.py", line 1002, in <module>
    main()
  File "/app/Pipeline/ipd4bq.py", line 981, in main
    write_to_bigquery(shortlisted, "shortlisted_secondary_patents_table")
  File "/app/Pipeline/ipd4bq.py", line 679, in write_to_bigquery
    _bq_retry_job(client, df, full_table_id, job_config=job_config)
    ^^^^^^^^^^^^^
NameError: name '_bq_retry_job' is not defined
