# Third-party notices

下列兩個子套件改寫自 browser-use 專案（https://github.com/browser-use/browser-use，MIT 授權），
在本 repo 內已改名為 `robot_agent.*`，並移除了用不到的部分（CLI、雲端同步、沙盒、遙測、瀏覽器整合等）：

- `robot_agent/core/`：agent 迴圈框架（觀察→思考→行動、訊息管理、動作註冊表、瀏覽器狀態的資料結構）
- `robot_agent/llm/`：各家 LLM 的呼叫包裝、訊息型別、結構化輸出的 schema 處理

其餘程式碼皆為本專案自行撰寫。依 MIT 授權要求保留下列版權與授權聲明：

```
MIT License

Copyright (c) 2024 Gregor Zunic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
