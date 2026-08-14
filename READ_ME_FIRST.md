# ShelfSight AI — Start Here

You do not need to know anything about programming to run this.

---

## To start the system

**Double-click `START.bat`**

That is the whole instruction. It will:

1. check that the two things it needs are installed,
2. install everything else by itself,
3. start the system,
4. open the dashboard in your browser.

**The first time takes 15–25 minutes** because it downloads the AI libraries
(about 2 GB). Leave the window open and let it finish — it prints what it is
doing at each step.

**Every time after that takes about 30 seconds.**

## To stop the system

**Double-click `STOP.bat`**

Your data is kept. Start it again whenever you like.

---

## What you need installed first

`START.bat` checks for these and tells you where to get them if they are
missing. You only ever install them once.

| Needed | Where to get it | Note |
|---|---|---|
| **Python 3.10 or newer** | https://www.python.org/downloads/ | On the first installer screen, **tick "Add python.exe to PATH"**. This matters — without it, nothing works. |
| **Node.js 18 or newer** | https://nodejs.org/ | Download the **LTS** version and accept every default. |

Restart your computer after installing them, then double-click `START.bat`.

---

## Where things are

Once it is running:

| What | Address |
|---|---|
| **The dashboard** — the app you actually use | http://localhost:3000 |
| The technical API documentation | http://localhost:8000/docs |

Both only work on this computer while the system is running.

---

## Folder layout

These two folders must sit **next to each other**. If you move one, `START.bat`
will not find the other and will tell you so.

```
Projects\
├── be\      <- the engine (this folder — START.bat is here)
└── fe\      <- the dashboard
```

---

## If something goes wrong

**"Python was not found"**
Python is not installed, or "Add python.exe to PATH" was not ticked during
installation. Reinstall it from python.org and tick that box.

**"The dashboard folder was not found"**
The `fe` folder is missing or is not beside this one. See the layout above.

**"Port 3000 is already in use"**
Something else on your computer is using that address. Double-click `STOP.bat`,
wait ten seconds, then `START.bat` again.

**The dashboard says "cannot reach the system"**
The engine takes 30–90 seconds to load its AI models when it first starts. Wait
a moment and refresh the page.

**It downloaded for a while and then failed**
Almost always the internet connection dropped. Just double-click `START.bat`
again — it keeps what it already downloaded and carries on from there.

**Nothing above helps**
Run `START.bat`, take a photograph of the window, and send it to whoever handed
you this project. The error text tells them what happened.

---

## What the system does

Point it at a photograph of a shop shelf and it will tell you:

- **which products are missing** from the shelf, and what that is costing you
- **whether products are in the right places** compared to your shelf plan
- **whether fruit and vegetables are fresh, ripening or spoiled**
- **what the expiry dates say** on packaging
- a **plain-language daily summary** of all of the above

All of it runs on this computer. Nothing is uploaded anywhere, there is no
account to create, and there are no ongoing costs.
