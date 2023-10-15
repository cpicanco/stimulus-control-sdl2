{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers.writerow;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

procedure WriteDataRow;

var
  BaseHeader,
  BlockName,
  LastTrialHeader,
  TrialHeader,
  TrialName,
  TrialData,
  TrialResult: string;
  ITIBegin,
  ITIEnd: Extended;

implementation

uses session.constants
   , timestamps
   , session.loggers
   , session.loggers.instances
   , session.pool;

const
  Tab = #9;

procedure InitializeBaseHeader;
begin
  ITIBegin := 0;
  ITIEnd := 0;
  BaseHeader := TLogger.Row([
    'Session.Trial.UID',
    'Session.Block.UID',
    'Session.Block.ID',
    'Session.Block.Trial.UID',
    'Session.Block.Trial.ID',
    'Session.Block.Name',
    'Session.Block.Trial.Name',
    'ITI.Begin',
    'ITI.End'],'');
  TrialHeader := '';
  LastTrialHeader := ' ';
end;

procedure WriteDataRow;
var
  LSaveData : TDataProcedure;
  ITIData, LData : string;
const
  DoNotApply = 'NA';
  EmptyName = '--------';
begin
  if TrialHeader <> LastTrialHeader then begin
    LData := TLogger.Row([BaseHeader, TrialHeader]);
  end;
  LastTrialHeader := TrialHeader;

  if BlockName.IsEmpty then begin
    BlockName := EmptyName;
  end;

  if TrialName.IsEmpty then begin
    TrialName := EmptyName;
  end;

  if Pool.Session.Trial.UID = 0 then begin
    ITIData := DoNotApply + Tab + TimestampToStr(0)
  end else begin
    ITIData :=
      TimestampToStr(ITIBegin) + Tab + TimestampToStr(TickCount - Pool.TimeStart);
  end;

  // write data
  LSaveData := GetSaveDataProc(LGData);
  LData := TLogger.Row([LData +
    (Pool.Session.Trial.UID+1).ToString,
    (Pool.Session.Block.UID + 1).ToString,
    (Pool.Session.Block.ID + 1).ToString,
    (Pool.Session.Block.Trial.UID + 1).ToString,
    (Pool.Session.Block.Trial.ID + 1).ToString,
    BlockName,
    TrialName,
    ITIData,
    TrialData]);
  LSaveData(LData);
end;

initialization
  InitializeBaseHeader;

end.

