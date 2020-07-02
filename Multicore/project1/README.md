# Scaffold for hw1

## What is it?

This is a scaffold for your hw1, including:
- a bunch of sample data to test against;
- the utilities to read-in the samples and write-out your results;
- a full project structure with CMakeLists.txt for writing your CUDA program.

More details can be found in the source code files.

## How to use it?

You typically just need to modify `sources/src/core.h` and `sources/src/core.cu`. Currently, these two source code files contain a demonstrating example but the example has nothing to do with the requirements of hw1.

When you are about to hand-in your solutions to hw1, make sure
- this README file has been replaced by your experiment report (the report doesn't need to be a Markdown file and besides, you are always suggested to include a PDF version of your report);
- your results against the sample data have been put in the `results` folder;
- the sample data file `data.bin` has been removed;
- the folder name has been changed to `{your name}-{your ID}` (curly brackets are not needed).

And finally, compress the whole folder as a `.zip` file or a `.7z` file, and send it to multicoresysu2020@163.com before 2020.07.05 23:59.