// Apps Script for the canonical "PlatinumQuest ML - Best Real Scores per Map" sheet.
// Deploy: Extensions > Apps Script > paste > Deploy > New deployment > type "Web app",
// Execute as: Me, Who has access: Anyone > Deploy > copy the Web app URL.
// POST JSON: {"token": "...", "best": [[map, score, falls, "YYYY-MM-DD"], ...],
//             "history": {"YYYY-MM-DD": {"MapName": score, ...}}}
// GET: returns the Best scores tab as JSON (for verification).
var TOKEN = 'pq-best-scores-2026';

function doGet(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var best = ss.getSheetByName('Best scores');
  var out = { best: best ? best.getDataRange().getDisplayValues() : null,
              tabs: ss.getSheets().map(function (s) { return s.getName(); }) };
  return ContentService.createTextOutput(JSON.stringify(out)).setMimeType(ContentService.MimeType.JSON);
}

function doPost(e) {
  var body = JSON.parse(e.postData.contents);
  if (body.token !== TOKEN) return ContentService.createTextOutput('bad token');
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var best = ss.getSheetByName('Best scores') || ss.getSheets()[0];
  var log = [];
  // --- Best scores tab: columns Map | Best real full-round score | Falls in that round | Date
  var data = best.getDataRange().getValues();
  var rowOf = {};
  for (var r = 1; r < data.length; r++) rowOf[String(data[r][0])] = r + 1;
  (body.best || []).forEach(function (row) {
    var map = row[0], score = row[1], falls = row[2], date = row[3];
    var r = rowOf[map];
    if (!r) { r = best.getLastRow() + 1; best.getRange(r, 1).setValue(map); rowOf[map] = r; }
    best.getRange(r, 2, 1, 3).setValues([[score, falls, date]]);
    best.getRange(r, 4).setNumberFormat('yyyy-mm-dd');
    log.push('best ' + map + ' = ' + score + ' (' + date + ')');
  });
  // --- History tab: Date | one column per map; one row per date, bests carried forward
  var hist = ss.getSheetByName('History');
  if (hist && body.history) {
    var hdata = hist.getDataRange().getValues();
    var header = hdata[0];
    var colOf = {};
    for (var c = 1; c < header.length; c++) if (header[c]) colOf[String(header[c])] = c + 1;
    Object.keys(body.history).sort().forEach(function (date) {
      var values = body.history[date];
      var r = -1;
      for (var i = 1; i < hdata.length; i++) {
        var cell = hdata[i][0];
        var txt = (cell instanceof Date) ? Utilities.formatDate(cell, Session.getScriptTimeZone(), 'yyyy-MM-dd') : String(cell);
        if (txt === date) { r = i + 1; break; }
      }
      if (r < 0) {
        // insert in date order: first row whose date is later, else append after the last date row
        r = 1;
        for (var i = 1; i < hdata.length; i++) {
          var cell = hdata[i][0];
          if (!cell) break;
          var txt = (cell instanceof Date) ? Utilities.formatDate(cell, Session.getScriptTimeZone(), 'yyyy-MM-dd') : String(cell);
          if (txt < date) r = i + 1;
        }
        hist.insertRowAfter(r); r = r + 1;
        hist.getRange(r, 1).setValue(date).setNumberFormat('yyyy-mm-dd');
        hdata = hist.getDataRange().getValues();
      }
      Object.keys(values).forEach(function (map) {
        var c = colOf[map];
        if (!c) { c = hist.getLastColumn() + 1; hist.getRange(1, c).setValue(map); colOf[map] = c; }
        hist.getRange(r, c).setValue(values[map]);
      });
      log.push('history ' + date + ' ' + Object.keys(values).length + ' maps');
    });
  }
  // --- clear: [{"tab": "History", "range": "M1"}, ...] empties cells (content only)
  (body.clear || []).forEach(function (c) {
    var sh = ss.getSheetByName(c.tab);
    if (sh) { sh.getRange(c.range).clearContent(); log.push('cleared ' + c.tab + '!' + c.range); }
  });
  // --- set: [{"tab": "History", "range": "A1", "value": "Date"}, ...] writes single cells
  (body.set || []).forEach(function (c) {
    var sh = ss.getSheetByName(c.tab);
    if (sh) { sh.getRange(c.range).setValue(c.value); log.push('set ' + c.tab + '!' + c.range); }
  });
  return ContentService.createTextOutput(JSON.stringify({ ok: true, log: log })).setMimeType(ContentService.MimeType.JSON);
}
