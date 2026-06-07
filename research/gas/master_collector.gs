/**
 * Master Collector — Google Apps Script
 *
 * Збирає метадані Google Forms і timestamp'и відповідей у master-spreadsheet.
 * Архітектура:
 *   - IDEMPOTENT per-form: кожна форма перезаписується повністю (видаляємо
 *     старі рядки за form_id, додаємо свіжі). Дублікати неможливі.
 *   - RESUMABLE: один run обробляє стільки форм, скільки встигне за 5 хв
 *     (з ~1 хв запасу до 6-хв квоти Apps Script). Cursor зберігається у
 *     scriptProperties; повторні запуски продовжують з місця зупинки.
 *   - BATCHED SHEET I/O: один read + один write за пакет, не per-form.
 *
 * Деплоймент:
 *   1. Apps Script Editor → новий проект з цим файлом.
 *   2. Project Settings → Script Properties → "MASTER_SHEET_ID" зі значенням ID
 *      master-spreadsheet'у.
 *   3. Services → enable "Drive API" (v3).
 *   4. Run `setupTrigger()` один раз — створить time-driven trigger на 4h.
 *   5. Для першого backfill'у: Run `collectAll()` повторно поки log не покаже
 *      "Pass complete". Кожен run опрацьовує батч і пише прогрес.
 *
 * Webhook (опційно):
 *   Deploy → New deployment → Web app → execute as self, anyone can access.
 *   POST ?formId=<id> — оновить лише цю форму (idempotent per-form).
 */

const CONFIG = {
  TRIGGER_HOURS: 4,
  MASTER_SHEET_ID: PropertiesService.getScriptProperties().getProperty("MASTER_SHEET_ID"),
  DRIVE_QUERY: 'mimeType="application/vnd.google-apps.form"',
  TIMESTAMP_FORMAT: "yyyy-MM-dd HH:mm:ss",
  // 5 хв з 6-хв квоти; останню хвилину тримаємо для commit'у sheet'ів і
  // запису state у scriptProperties (вони не миттєві на великих об'ємах).
  BATCH_TIME_LIMIT_MS: 5 * 60 * 1000,
  SHEETS: {
    TIMESTAMPS: {
      name: "Timestamps",
      header: ["Form ID", "Response Index", "Timestamp"],
    },
    FORMS: {
      name: "Forms",
      header: [
        "Form ID", "Form Link", "Name", "Title", "Owner", "Creation Date",
        "First Response Date", "Last Response Date", "Total Responses",
        "Connected Sheet ID", "Sheet Link", "Number of Sections",
        "Number of Questions", "Accepting", "Not Accepting Message",
        "Form Description",
      ],
    },
    ERRORS: {
      name: "Errors",
      header: ["Form ID", "Error Message", "Timestamp"],
    },
  },
};

const STATE_KEYS = {
  FILE_LIST: "collectAll_fileList",
  CURSOR: "collectAll_cursor",
};

/* =========================================================================
 * Public entry points
 * ========================================================================= */

/**
 * Resumable bulk collection: обробляє стільки форм, скільки встигне за
 * BATCH_TIME_LIMIT_MS. Cursor + cached file list persistяться у
 * scriptProperties між run'ами.
 *
 * Викликати поки log не покаже "Pass complete" — приблизно 2-3 рази для
 * 179 форм залежно від розміру.
 */
function collectAll() {
  const startTime = Date.now();
  const ss = SpreadsheetApp.openById(CONFIG.MASTER_SHEET_ID);
  const tsSheet = ensureSheet(ss, CONFIG.SHEETS.TIMESTAMPS);
  const formsSheet = ensureSheet(ss, CONFIG.SHEETS.FORMS);
  const errSheet = ensureSheet(ss, CONFIG.SHEETS.ERRORS);
  const props = PropertiesService.getScriptProperties();

  // Resume cached file list across runs; коли пас завершено — старт нового.
  let fileList = readJsonProp(props, STATE_KEYS.FILE_LIST);
  let cursor = parseInt(props.getProperty(STATE_KEYS.CURSOR) || "0", 10);
  if (!fileList) {
    fileList = listAllForms();
    props.setProperty(STATE_KEYS.FILE_LIST, JSON.stringify(fileList));
    cursor = 0;
    props.setProperty(STATE_KEYS.CURSOR, "0");
    Logger.log(`New pass started: ${fileList.length} forms.`);
  }

  const newTimestamps = [];
  const newMetadata = [];
  const formIdsProcessed = [];
  const errors = [];

  while (cursor < fileList.length) {
    if (Date.now() - startTime > CONFIG.BATCH_TIME_LIMIT_MS) break;

    const entry = fileList[cursor];
    try {
      // Drive.Files.list повертає plain JSON без методів DriveApp.File,
      // тому беремо file окремим викликом.
      const file = DriveApp.getFileById(entry.id);
      const form = FormApp.openById(entry.id);
      newMetadata.push(buildFormMetadata(file, form));
      const tsRows = buildTimestampRows(entry.id, form);
      for (const r of tsRows) newTimestamps.push(r);
      formIdsProcessed.push(entry.id);
    } catch (e) {
      errors.push([entry.id, e.toString(), formatNow()]);
    }
    cursor++;
  }

  // Atomic batch commit: один read + один write на Timestamps, незалежно
  // від кількості форм у пакеті.
  if (formIdsProcessed.length > 0) {
    batchReplaceTimestamps(tsSheet, formIdsProcessed, newTimestamps);
  }
  for (const meta of newMetadata) upsertFormRow(formsSheet, meta);
  if (errors.length > 0) appendRows(errSheet, errors);

  if (cursor >= fileList.length) {
    props.deleteProperty(STATE_KEYS.FILE_LIST);
    props.deleteProperty(STATE_KEYS.CURSOR);
    Logger.log(`Pass complete: ${fileList.length} forms processed.`);
  } else {
    props.setProperty(STATE_KEYS.CURSOR, cursor.toString());
    const pct = ((cursor / fileList.length) * 100).toFixed(1);
    Logger.log(
      `Partial pass: ${cursor}/${fileList.length} (${pct}%). Re-run collectAll to continue.`
    );
  }
}

/**
 * Per-form refresh: idempotent оновлення однієї форми.
 * Видаляє існуючі рядки [formId, *, *] з Timestamps і додає свіжі.
 * Forms-рядок upsert'иться.
 */
function refreshForm(formId) {
  const ss = SpreadsheetApp.openById(CONFIG.MASTER_SHEET_ID);
  const tsSheet = ensureSheet(ss, CONFIG.SHEETS.TIMESTAMPS);
  const formsSheet = ensureSheet(ss, CONFIG.SHEETS.FORMS);
  const errSheet = ensureSheet(ss, CONFIG.SHEETS.ERRORS);

  try {
    const file = DriveApp.getFileById(formId);
    const form = FormApp.openById(formId);
    upsertFormRow(formsSheet, buildFormMetadata(file, form));
    batchReplaceTimestamps(tsSheet, [formId], buildTimestampRows(formId, form));
    return { success: true, formId: formId };
  } catch (e) {
    appendRows(errSheet, [[formId, e.toString(), formatNow()]]);
    return { success: false, formId: formId, error: e.toString() };
  }
}

/**
 * Manual reset: скинути cursor + кеш file list, щоб наступний collectAll
 * почав свіжий повний пас. Викликати, якщо щось пішло не так і хочеш
 * почати з нуля.
 */
function resetCollectionState() {
  const props = PropertiesService.getScriptProperties();
  props.deleteProperty(STATE_KEYS.FILE_LIST);
  props.deleteProperty(STATE_KEYS.CURSOR);
  Logger.log("Collection state reset. Next collectAll() starts fresh pass.");
}

/* =========================================================================
 * Webhook
 * ========================================================================= */

function doPost(e) {
  const params = (e && e.parameter) || {};
  if (!params.formId) {
    return jsonResponse({ success: false, error: "Missing 'formId' parameter." });
  }
  return jsonResponse(refreshForm(params.formId));
}

function doGet(_e) {
  return jsonResponse({
    service: "master-collector",
    routes: ["POST ?formId=<id>"],
  });
}

/* =========================================================================
 * Form processing
 * ========================================================================= */

/**
 * @param {DriveApp.File} file - повноцінний DriveApp.File (НЕ Drive.Files.list result).
 * @param {FormApp.Form} form
 */
function buildFormMetadata(file, form) {
  const tz = Session.getScriptTimeZone();
  const responses = form.getResponses();
  const n = responses.length;
  const fmt = (d) => Utilities.formatDate(d, tz, CONFIG.TIMESTAMP_FORMAT);

  let destSheetId = "", sheetLink = "";
  try {
    destSheetId = form.getDestinationId() || "";
    if (destSheetId) sheetLink = `https://docs.google.com/spreadsheets/d/${destSheetId}/edit`;
  } catch (_) { /* form has no destination */ }

  // getOwner() може повернути null для Shared Drive файлів.
  let ownerEmail = "";
  try {
    const owner = file.getOwner();
    if (owner) ownerEmail = owner.getEmail();
  } catch (_) { /* no permission to owner */ }

  const sections = form.getItems(FormApp.ItemType.PAGE_BREAK).length;
  const accepting = form.isAcceptingResponses();
  const formId = file.getId();

  return [
    formId,
    `https://docs.google.com/forms/d/${formId}/edit`,
    file.getName(),
    form.getTitle(),
    ownerEmail,
    fmt(file.getDateCreated()),
    n > 0 ? fmt(responses[0].getTimestamp()) : "",
    n > 0 ? fmt(responses[n - 1].getTimestamp()) : "",
    n,
    destSheetId,
    sheetLink,
    sections,
    form.getItems().length - sections,
    accepting ? "Yes" : "No",
    !accepting ? (form.getCustomClosedFormMessage() || "") : "",
    form.getDescription(),
  ];
}

function buildTimestampRows(formId, form) {
  const tz = Session.getScriptTimeZone();
  const responses = form.getResponses();
  const rows = new Array(responses.length);
  for (let i = 0; i < responses.length; i++) {
    rows[i] = [
      formId,
      i + 1,
      Utilities.formatDate(responses[i].getTimestamp(), tz, CONFIG.TIMESTAMP_FORMAT),
    ];
  }
  return rows;
}

/* =========================================================================
 * Sheet helpers
 * ========================================================================= */

function ensureSheet(ss, spec) {
  let sheet = ss.getSheetByName(spec.name);
  if (!sheet) {
    sheet = ss.insertSheet(spec.name);
    sheet.appendRow(spec.header);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

/**
 * Повна заміна вмісту нижче header'а на нові рядки.
 */
function rewriteSheet(sheet, rows) {
  const lastRow = sheet.getLastRow();
  if (lastRow > 1) {
    sheet.getRange(2, 1, lastRow - 1, sheet.getLastColumn()).clearContent();
  }
  if (rows.length > 0) {
    sheet.getRange(2, 1, rows.length, rows[0].length).setValues(rows);
  }
}

function appendRows(sheet, rows) {
  if (rows.length === 0) return;
  const startRow = sheet.getLastRow() + 1;
  sheet.getRange(startRow, 1, rows.length, rows[0].length).setValues(rows);
}

/**
 * Upsert одного рядка у Forms за Form ID (стовпець 1).
 */
function upsertFormRow(sheet, row) {
  const lastRow = sheet.getLastRow();
  if (lastRow >= 2) {
    const ids = sheet.getRange(2, 1, lastRow - 1, 1).getValues();
    for (let i = 0; i < ids.length; i++) {
      if (ids[i][0] === row[0]) {
        sheet.getRange(i + 2, 1, 1, row.length).setValues([row]);
        return;
      }
    }
  }
  appendRows(sheet, [row]);
}

/**
 * Batch-replace: видалити всі рядки з formId ∈ formIds і дописати newRows.
 * Один read + один write незалежно від кількості форм у пакеті.
 */
function batchReplaceTimestamps(sheet, formIds, newRows) {
  const idSet = {};
  for (const id of formIds) idSet[id] = true;

  const lastRow = sheet.getLastRow();
  let kept = [];
  if (lastRow > 1) {
    const all = sheet.getRange(2, 1, lastRow - 1, sheet.getLastColumn()).getValues();
    for (const r of all) {
      if (!idSet[r[0]]) kept.push(r);
    }
  }
  rewriteSheet(sheet, kept.concat(newRows));
}

/* =========================================================================
 * Drive listing
 * ========================================================================= */

function listAllForms() {
  const files = [];
  let pageToken = null;
  do {
    const resp = Drive.Files.list({
      q: CONFIG.DRIVE_QUERY,
      fields: "nextPageToken, files(id, name)",
      pageToken: pageToken,
      pageSize: 1000,
    });
    if (resp.files) for (const f of resp.files) files.push(f);
    pageToken = resp.nextPageToken;
  } while (pageToken);
  return files;
}

/* =========================================================================
 * Misc
 * ========================================================================= */

function jsonResponse(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(
    ContentService.MimeType.JSON
  );
}

function formatNow() {
  return Utilities.formatDate(
    new Date(),
    Session.getScriptTimeZone(),
    CONFIG.TIMESTAMP_FORMAT
  );
}

function readJsonProp(props, key) {
  const raw = props.getProperty(key);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
}

/* =========================================================================
 * Trigger setup (run once manually)
 * ========================================================================= */

function setupTrigger() {
  // Видалити обсолетні handler-імена від попередніх версій скрипта.
  const obsolete = {
    collectFormTimestamps: 1,
    collectFormTimestampsBatch: 1,
  };
  let removed = 0;
  for (const t of ScriptApp.getProjectTriggers()) {
    if (obsolete[t.getHandlerFunction()]) {
      ScriptApp.deleteTrigger(t);
      removed++;
    }
  }
  // Не дублюємо collectAll, якщо вже існує clock-trigger.
  const hasCollectAll = ScriptApp.getProjectTriggers().some(
    (t) => t.getHandlerFunction() === "collectAll"
  );
  if (!hasCollectAll) {
    ScriptApp.newTrigger("collectAll")
      .timeBased()
      .everyHours(CONFIG.TRIGGER_HOURS)
      .create();
  }
  Logger.log(`Trigger setup: removed ${removed} obsolete; collectAll periodic active.`);
}
