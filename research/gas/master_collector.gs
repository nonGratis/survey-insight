/**
 * Master Collector — Google Apps Script
 *
 * Збирає метадані Google Forms і timestamp'и відповідей у master-spreadsheet.
 * Архітектура: IDEMPOTENT bulk rewrite — кожен run повністю перезаписує
 * Timestamps і Forms аркуші. Жодних cursor'ів у scriptProperties, жодних
 * дублікатів за побудовою.
 *
 * Деплоймент:
 *   1. Apps Script Editor → новий проект з цим файлом.
 *   2. Project Settings → Script Properties → додати "MASTER_SHEET_ID"
 *      зі значенням ID master-spreadsheet'у.
 *   3. Services → enable "Drive API" (v3).
 *   4. Run `setupTrigger()` один раз — створить time-driven trigger на 4h.
 *
 * Webhook (опційно):
 *   Deploy → New deployment → Web app → execute as self, anyone can access.
 *   POST /exec?formId=<id> — оновить лише цю форму (idempotent per-form).
 */

const CONFIG = {
  TRIGGER_HOURS: 4,
  MASTER_SHEET_ID: PropertiesService.getScriptProperties().getProperty("MASTER_SHEET_ID"),
  DRIVE_QUERY: 'mimeType="application/vnd.google-apps.form"',
  TIMESTAMP_FORMAT: "yyyy-MM-dd HH:mm:ss",
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

/* =========================================================================
 * Public entry points
 * ========================================================================= */

/**
 * Bulk collection: повний rewrite Timestamps + Forms для всіх accessible форм.
 * Викликається трігером (кожні TRIGGER_HOURS) або вручну.
 */
function collectAll() {
  const ss = SpreadsheetApp.openById(CONFIG.MASTER_SHEET_ID);
  const tsSheet = ensureSheet(ss, CONFIG.SHEETS.TIMESTAMPS);
  const formsSheet = ensureSheet(ss, CONFIG.SHEETS.FORMS);
  const errSheet = ensureSheet(ss, CONFIG.SHEETS.ERRORS);

  const allTimestamps = [];
  const allMetadata = [];
  const errors = [];

  const files = listAllForms();
  for (const file of files) {
    try {
      const form = FormApp.openById(file.id);
      allMetadata.push(buildFormMetadata(file, form));
      const tsRows = buildTimestampRows(file.id, form);
      for (const row of tsRows) allTimestamps.push(row);
    } catch (e) {
      errors.push([file.id, e.toString(), formatNow()]);
    }
  }

  // Idempotent rewrites: повна заміна вмісту нижче header'а.
  rewriteSheet(formsSheet, allMetadata);
  rewriteSheet(tsSheet, allTimestamps);

  // Errors — append-only (хочемо історію збоїв).
  if (errors.length > 0) appendRows(errSheet, errors);
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
    replaceTimestampsFor(tsSheet, formId, buildTimestampRows(formId, form));

    return { success: true, formId: formId };
  } catch (e) {
    appendRows(errSheet, [[formId, e.toString(), formatNow()]]);
    return { success: false, formId: formId, error: e.toString() };
  }
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

function doGet(e) {
  return jsonResponse({
    service: "master-collector",
    routes: ["POST ?formId=<id>"],
  });
}

/* =========================================================================
 * Form processing
 * ========================================================================= */

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

  const sections = form.getItems(FormApp.ItemType.PAGE_BREAK).length;
  const accepting = form.isAcceptingResponses();
  const formId = file.getId();

  return [
    formId,
    `https://docs.google.com/forms/d/${formId}/edit`,
    file.getName(),
    form.getTitle(),
    file.getOwner() ? file.getOwner().getEmail() : "",
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
 * Sheet helpers (idempotent operations)
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
 * Idempotent: повторний виклик з тими самими даними дає той самий sheet.
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
  const data = sheet.getRange(2, 1, Math.max(sheet.getLastRow() - 1, 0), 1).getValues();
  for (let i = 0; i < data.length; i++) {
    if (data[i][0] === row[0]) {
      sheet.getRange(i + 2, 1, 1, row.length).setValues([row]);
      return;
    }
  }
  appendRows(sheet, [row]);
}

/**
 * Видалити всі Timestamp-рядки даної форми і дописати нові.
 * Робиться через read → filter → rewrite, щоб уникнути deleteRow у циклі.
 */
function replaceTimestampsFor(sheet, formId, newRows) {
  const lastRow = sheet.getLastRow();
  let kept = [];
  if (lastRow > 1) {
    const all = sheet.getRange(2, 1, lastRow - 1, sheet.getLastColumn()).getValues();
    kept = all.filter((r) => r[0] !== formId);
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

/* =========================================================================
 * Trigger setup (run once manually)
 * ========================================================================= */

function setupTrigger() {
  // Видалити будь-які старі тригери на collectAll/collectFormTimestamps/Batch.
  const obsolete = new Set(["collectAll", "collectFormTimestamps", "collectFormTimestampsBatch"]);
  for (const t of ScriptApp.getProjectTriggers()) {
    if (obsolete.has(t.getHandlerFunction())) ScriptApp.deleteTrigger(t);
  }
  ScriptApp.newTrigger("collectAll")
    .timeBased()
    .everyHours(CONFIG.TRIGGER_HOURS)
    .create();
}
