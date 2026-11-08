# AI use disclosure

**Tool used:** Claude (Anthropic), as a chat assistant.

**How it was used:**
- Claude wrote the code for all four parts (`partA` to `partD`), the tests, `run_all.sh` and `make_submission.sh` from the assignment specification, and tested them on its own sandbox before I ran them.
- After I ran `bash run_all.sh` on my laptop, Claude drafted `REPORT.md` from the output files of that run.
- I used it to fix setup problems (conda, GitHub email privacy).

**What I did:** I ran everything on my own laptop. Every number in `REPORT.md` and in the output files
(`partA/*.json`, `partB/*.txt`, `partC/verify_output.txt`, `partD/smoke_output.txt`) comes from that run. No result was typed in
by hand or invented. I reviewed the code and the report before submitting.

**Impact:** the design and implementation came largely from the assistant. My part was running and checking the
pipeline, reviewing the results, and making sure I can explain each part.
