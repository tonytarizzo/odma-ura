- Adapt decoding setup for v3 and v4
- Go over additional complexities to be added in detail, consequences to assumptions, simplifications in v2
- Add per iteration logs for the saved results files, so that progress can be tracked as more complex setups need more computation

- implement an alternative of v1 that doesnt throw away cross variance terms in step 1, this is to evaluate how much of an effect those have, even if it increases computation a lot