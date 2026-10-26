# The approved design (plan 34, phase R)

The mockups the owner approved for the app's redesign (October 2026), kept as drawn: HTML and CSS at a phone's size
(390 x 844) and a landscape tablet's (1280 x 800). They are the reference the app's pages follow; the app itself is plain
MAUI XAML (`src/Idrak.Maui`), its colours, sizes and icons in the theme's dictionaries (`src/Idrak.Maui/Theme`).
Open a file in a browser to see it (the fonts are Inter and JetBrains Mono, bundled with the app).

## Phone

| File | Screen |
|---|---|
| `base.css` | The shared palette (dark), the bottom navigation bar, chips, icon buttons and section labels |
| `landing.html` | Home: the header with the source commit and the device chip, the carousel of four slides, "This phone, measured" and the "Try it" tiles |
| `shell.html` | All labs, the bottom sheet over Home: search, grouped by Language, Voice, Vision and This phone, planned labs dashed. Its blurred background is `landing.html` rendered (`landing.png`, not kept) |
| `record.html` | Voice, Transcribe: the model, language and cleaning chips, the live waveform card, the timestamped transcript, the record button with File and Tune |
| `chat.html` | Chat: the model card, the user's gradient bubbles, the answer with "Thought for", lists and code, the meta row, the streaming answer with its rate and Stop, the composer |

## Tablet (landscape)

| File | Screen |
|---|---|
| `tab.css`, `tab.js` | The side rail (logo; Home, Chat, Voice, Vision; All labs and Settings at the foot) and the status bar, shared by the tablet pages |
| `icons.js` | The icon set (24-unit stroke icons): the source of `src/Idrak.Maui/Theme/IdrakIcons.xaml` |
| `t-landing.html` | Home in two columns: the carousel and "Try it" on the left, the measured device and the models on the right |
| `t-shell.html` | All labs as a page: search, filter chips (All, Ready, Planned, Needs the microphone, Needs the camera), a column a group |
| `t-record.html` | Voice, Transcribe: the controls on the left (waveform, measures, the cleaning pipeline), the transcript on the right |
| `t-chat.html` | Chat in three panes: the conversations with search, the chat, the inspector (live tokens per second, memory, quick settings) |
