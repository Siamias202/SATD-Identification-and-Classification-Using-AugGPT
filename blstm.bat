@echo off
REM ============================================================
REM  BiLSTM Identification Runner - Windows 11
REM  Runs experiments for all artifacts (CC, IS, PS, CM)
REM  Script 2 of 3 from SATD Replication
REM ============================================================

REM ---- Configuration ----
set PROCESSED_DIR=.\processed
set GLOVE_PATH=.\glove.6B\glove.6B.300d.txt
set EMBED_DIM=300
set MAX_LEN=128
set BATCH_SIZE=32
set LR=0.001
set MAX_EPOCHS=50
set PATIENCE=5

REM ---- Run for Code Comments (CC) ----
echo ================================================
echo Running BiLSTM Identification for Artifact: CC
echo ================================================
python 02_bilstm_identification.py ^
    --processed_dir %PROCESSED_DIR% ^
    --artifact CC ^
    --glove_path %GLOVE_PATH% ^
    --embed_dim %EMBED_DIM% ^
    --max_len %MAX_LEN% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --max_epochs %MAX_EPOCHS% ^
    --patience %PATIENCE%

REM ---- Run for Issue Reports (IS) ----
echo ================================================
echo Running BiLSTM Identification for Artifact: IS
echo ================================================
python 02_bilstm_identification.py ^
    --processed_dir %PROCESSED_DIR% ^
    --artifact IS ^
    --glove_path %GLOVE_PATH% ^
    --embed_dim %EMBED_DIM% ^
    --max_len %MAX_LEN% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --max_epochs %MAX_EPOCHS% ^
    --patience %PATIENCE%

REM ---- Run for Pull Requests (PS) ----
echo ================================================
echo Running BiLSTM Identification for Artifact: PS
echo ================================================
python 02_bilstm_identification.py ^
    --processed_dir %PROCESSED_DIR% ^
    --artifact PS ^
    --glove_path %GLOVE_PATH% ^
    --embed_dim %EMBED_DIM% ^
    --max_len %MAX_LEN% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --max_epochs %MAX_EPOCHS% ^
    --patience %PATIENCE%

REM ---- Run for Commit Messages (CM) ----
echo ================================================
echo Running BiLSTM Identification for Artifact: CM
echo ================================================
python 02_bilstm_identification.py ^
    --processed_dir %PROCESSED_DIR% ^
    --artifact CM ^
    --glove_path %GLOVE_PATH% ^
    --embed_dim %EMBED_DIM% ^
    --max_len %MAX_LEN% ^
    --batch_size %BATCH_SIZE% ^
    --lr %LR% ^
    --max_epochs %MAX_EPOCHS% ^
    --patience %PATIENCE%

echo ============================================================
echo All BiLSTM experiments completed!
echo ============================================================
pause
