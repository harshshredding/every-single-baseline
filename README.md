# NER framework
This framework should allow you to efficiently train a variety of NER models on your data. Moreover, it makes it easy to inspect the mistakes that the trained model is making.

# Steps to train on your data
1. Create 3 files: 
    - `input.json`: 
    - `annotation.json`:
    - `types.txt`:
2. Write tests to ensure the consistency of the above three files.
3. Create gate input file called `gate_input.bdocjs` to inspect your data in GATE Developer. Inspect your data _carefully_ and try to spot bugs in your importing the data.
4. Train your model.
5. Inspect the mistakes your model made by opening `mistakes.bdocjs` in GATE developer.
6. Improve your model and go back to step 4.


Set PYTHONPATH to root directory.