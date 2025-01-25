{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers.writerow;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Math, session.loggers.types;

procedure WriteDataRow;

var
  SaveData : TDataProcedure = nil;
  BlockName : string;
  TrialName : string;
  //ITIBegin  : Float;
  //ITIEnd    : Float;

procedure AppendToTrialHeader(AHeader : string);
procedure AppendToTrialData(AData : string);
procedure InitializeBaseHeader;

implementation

uses session.constants
   , timestamps
   , timestamps.methods
   , session.loggers
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
  LData := TLogger.Row([LData +
    ClockMonotonic.ToString,
    (Pool.Session.Trial.UID + 1).ToString,
    (Pool.Session.Block.UID + 1).ToString,
    (Pool.Session.Block.Trial.UID + 1).ToString,
    (Pool.Session.Block.ID + 1).ToString,
    (Pool.Session.Block.Trial.ID + 1).ToString,
    BlockName,
    TrialName,
    TrialData]);
  SaveData(LData);
  TrialData := '';
  TrialHeader := '';
end;

end.

