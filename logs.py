Traceback (most recent call last):
  File "/app/Pipeline/ipd2bq.py", line 1679, in <module>
    main()
  File "/app/Pipeline/ipd2bq.py", line 1625, in main
    results = asyncio.run(_score_all())
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 190, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/asyncio/base_events.py", line 654, in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/app/Pipeline/ipd2bq.py", line 1621, in _score_all
    await asyncio.gather(*tasks)
  File "/app/Pipeline/ipd2bq.py", line 1612, in _process_one
    result = await score_patent_async(
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/Pipeline/ipd2bq.py", line 887, in score_patent_async
    all_chunks = await loop.run_in_executor(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/Pipeline/ipd2bq.py", line 672, in fetch_all_chunks_for_patent
    from cog.indexer import get_filename_collection_map_sync
ImportError: cannot import name 'get_filename_collection_map_sync' from 'cog.indexer' (/app/Pipeline/cog/indexer.py)
