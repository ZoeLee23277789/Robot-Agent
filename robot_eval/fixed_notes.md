# Human-reviewed notes for evaluation (fixed condition)

這份檔案是評估用的固定先驗，人工審核過，評估執行時不會被改寫（RobotAgent 用 persist_notes=False
建構，remember() 照常回應但不寫回這個檔案）。所有題目、所有被比較的模型都拿到完全一樣的這幾條，
不會因為某一題剛好先跑而佔到別人沒有的提示。改動這份檔案是人的決定，不是 agent 自己寫的。

- Use arm pose (x:90, y:120) to search the floor far ahead, and arm pose (x:180, y:30) to see the floor directly in front of the gripper.
