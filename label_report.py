Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/Pipeline/cog/main.py", line 178, in <module>
    main()
  File "/app/Pipeline/cog/main.py", line 169, in main
    result = asyncio.run(
             ^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/base_events.py", line 654, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/app/Pipeline/cog/main.py", line 112, in run_single
    result = await get_dimension_i_patent_data(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/Pipeline/cog/tools.py", line 271, in get_dimension_i_patent_data
    timeline = await fetch_clinical_timeline(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/Pipeline/cog/phase_fetcher.py", line 633, in fetch_clinical_timeline
    bq_val = bq_geography[geo]
             ~~~~~~~~~~~~^^^^^
KeyError: 'EU'
