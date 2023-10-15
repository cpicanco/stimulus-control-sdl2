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
  BlockName : string;
  TrialName : string;
  //ITIBegin  : Extended;
  //ITIEnd    : Extended;

procedure AppendToTrialHeader(AHeader : string);
procedure AppendToTrialData(AData : string);

implementation

uses session.constants
   , timestamps
   , session.loggers
   , session.loggers.instances
   , session.pool;

var
  BaseHeader,
  LastTrialHeader,
  TrialHeader,
  TrialData : string;

const
  Tab = #9;

procedure Append(var ALeft: string; ARight  : string);
begin
  if ALeft.IsEmpty then begin
    ALeft := ARight;
  end else begin
    ALeft := String.Join(Tab, [ALeft, ARight]);
  end;
end;

procedure AppendToTrialHeader(AHeader: string);
begin
  Append(TrialHeader, AHeader);
end;

procedure AppendToTrialData(AData: string);
begin
  Append(TrialData, AData);
end;

procedure InitializeBaseHeader;
begin
  BaseHeader := String.Join(Tab, [
    'Report.Timestamp',
    'Session.Trial.UID',
    'Session.Block.UID',
    'Session.Block.Trial.UID',
    'Session.Block.ID',
    'Session.Block.Trial.ID',
    'Session.Block.Name',
    'Session.Block.Trial.Name']);
  TrialHeader := '';
  LastTrialHeader := ' ';
end;

procedure WriteDataRow;
var
  LSaveData : TDataProcedure;
  LData : string;
const
  EmptyName = '--------';
begin
  if TrialHeader <> LastTrialHeader then begin
    LData := TLogger.Row([BaseHeader, TrialHeader]);
    LastTrialHeader := TrialHeader;
  end;

  if BlockName.IsEmpty then begin
    BlockName := EmptyName;
  end;

  if TrialName.IsEmpty then begin
    TrialName := EmptyName;
  end;

  // write data
  LSaveData := GetSaveDataProc(LGData);
  LData := TLogger.Row([LData +
    TimestampToStr(TickCount - Pool.TimeStart),
    (Pool.Session.Trial.UID + 1).ToString,
    (Pool.Session.Block.UID + 1).ToString,
    (Pool.Session.Block.Trial.UID + 1).ToString,
    (Pool.Session.Block.ID + 1).ToString,
    (Pool.Session.Block.Trial.ID + 1).ToString,
    BlockName,
    TrialName,
    TrialData]);
  LSaveData(LData);
  TrialData := '';
end;

initialization
  InitializeBaseHeader;

end.

