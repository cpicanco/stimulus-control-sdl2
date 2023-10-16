{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters;

{$mode objfpc}{$H+}

{$ModeSwitch advancedrecords}

interface

uses session.configuration, session.counters.consecutive, session.counters.all;

type

  { TCounters }

  TCounters = record
  public
    Subject : Word;
    Session : TSessionCounters;
    Block : TBlockCounters;
    Trial : TTrialCounters;
    function EndTrial(ANextTrial: SmallInt) : Boolean;
    function EndBlock(ANextBlock : SmallInt): Boolean;
    procedure BeforeEndTrial;
    procedure BeforeEndBlock;
    procedure Hit;
    procedure Miss;
    procedure None;
    procedure RandomEvent;
    procedure Reset;
    procedure BeforeBeginSession;
    procedure EndSession;
  end;

var
  Counters : TCounters;

implementation

uses Classes, SysUtils
  , sdl.app.output
  , sdl.app.grids
  , session.loggers.writerow
  , session.pool
  , session.configurationfile
  ;

{ TCounterManager }

function GetSubjectIDFromFile : Word;
var
  LStringList : TStringList;
begin
  LStringList := TStringList.Create;
  try
    try
      LStringList.LoadFromFile(Pool.BaseFileName+'ID');
    except
      on EFileNotFoundException do

    end;
    Result := LStringList[0].ToInteger;
  finally
    LStringList.Free;
  end;
end;

procedure TCounters.BeforeBeginSession;
begin
  Subject := GetSubjectIDFromFile;
  Session := TSessionCounters.Create;
  Session.Reset;
  Block := Session.Block;
  Trial := Block.Trial;

  Pool.Session := Session;
  Pool.Block   := Block;
  Pool.Trial   := Trial;

  Trial.ID := ConfigurationFile.StartAt.Trial;
  Block.ID := ConfigurationFile.StartAt.Block;
end;

function TCounters.EndBlock(ANextBlock: SmallInt) : Boolean;
begin
  Result := True;
  Session.Block.Events.Invalidate;

  if ANextBlock = Block.ID then begin
    Session.Block.NextConsecutive;
  end else begin
    Session.Block.ResetConsecutive;
  end;

  Session.NextBlockID(ANextBlock);

  if (ANextBlock > -1) and (ANextBlock < Length(Session.Tree.Block)) then begin
    Result := True;
  end else begin
    Result := False;
  end;
end;

procedure TCounters.BeforeEndTrial;
begin
  Session.Tree.Block[Block.ID].Trial[Trial.ID].Increment;
  AppendToTrialData(Session.Block.Trial.Events.Last);
  AppendToTrialData(Session.Trial.Events.ToData);
  AppendToTrialData(Grid.ToData);
  WriteDataRow;
end;

procedure TCounters.BeforeEndBlock;
begin
  Session.Tree.Block[Block.ID].Increment;
end;

procedure TCounters.EndSession;
begin
  Session.Free;
end;

function TCounters.EndTrial(ANextTrial: SmallInt) : Boolean;
begin
  Session.Block.Trial.Events.Invalidate;

  if ANextTrial = Trial.ID then begin
    Session.Trial.NextConsecutive;
    Session.Block.Trial.NextConsecutive;
  end else begin
    Session.Trial.ResetConsecutive;
    Session.Block.Trial.ResetConsecutive;
  end;

  Session.NextTrialID(ANextTrial);

  if (ANextTrial > -1) and
     (ANextTrial < Length(Session.Tree.Block[Block.ID].Trial)) then begin
    Result := True;
  end else begin
    Result := False;
  end;
end;

procedure TCounters.Hit;
begin
  Session.Events.Hit;
  Session.Trial.Events.Hit;
  Session.Block.Events.Hit;
  Session.Block.Trial.Events.Hit;
end;

procedure TCounters.Miss;
begin
  Session.Events.Miss;
  Session.Trial.Events.Miss;
  Session.Block.Events.Miss;
  Session.Block.Trial.Events.Miss;
end;

procedure TCounters.None;
begin
  Session.Events.None;
  Session.Trial.Events.None;
  Session.Block.Events.None;
  Session.Block.Trial.Events.None;
end;

procedure TCounters.RandomEvent;
begin
  case Random(3) of
    0 : Hit;
    1 : Miss;
    2 : None;
  end;
end;

procedure TCounters.Reset;
begin
  Session.Events.Reset;
  Session.Trial.Events.Reset;
  Session.Block.Events.Reset;
  Session.Block.Trial.Events.Reset;
end;

end.
