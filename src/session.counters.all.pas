{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.all;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.output
  , session.configurationfile
  , session.counters.hierarquical
  , session.counters.consecutive
  , session.counters.mutuallyexclusive;

type
    TTrialID = Word;
    TBlockID = Word;

    { TCustomUIDCounter }

    TCustomUIDCounter = class(TUIDCounter)
    protected
      FCustomName : string;
      FEvents : TMutuallyExclusiveCounters;
      function ToString : string; override;
    public
      constructor Create;
      destructor Destroy; override;
      procedure Next; override;
      procedure NextConsecutive; override;
      procedure NextID(AValue : TTrialID); virtual; abstract;
      property Events : TMutuallyExclusiveCounters read FEvents;
      property CustomName : string read FCustomName write FCustomName;
    end;

    { TTrialCounters }

    TTrialCounters = class(TCustomUIDCounter)
    protected
      function ToString : string; override;
    public
      ID : TTrialID; static;
      procedure Invalidate; override;
      procedure NextID(AValue : TTrialID); override;
    end;

    { TBlockCounters }

    TBlockCounters = class(TCustomUIDCounter)
    strict private
      FTrial : TTrialCounters;
    protected
      function GetTrials: Word; overload;
      function ToString : string; override;
    public
      ID : TBlockID; static;
      constructor Create;
      destructor Destroy; override;
      procedure Invalidate; override;
      procedure Next; override;
      procedure NextID(AValue : TTrialID); override;
      procedure ResetConsecutive; override;
      property Trial : TTrialCounters read FTrial write FTrial;
      property Trials : Word read GetTrials;
    end;

    { TSessionCounters }

    TSessionCounters = class(TCustomUIDCounter)
    strict private
      FTree : TIDSessionCounter;
      FBlock : TBlockCounters;
      FTrial : TTrialCounters;
    protected
      function GetTrials : Word;
      function GetBlocks : Word;
      procedure NextTrial;

    public
      ID : Word; static;
      constructor Create;
      destructor Destroy; override;
      procedure Reset; override;
      procedure NextID(AValue : TTrialID); override;
      procedure NextTrialConsecutive;
      //procedure ResetTrialConsecutive;
      procedure NextTrialID(ATrialID : TTrialID);
      //procedure NextBlockConsecutive;
      //procedure ResetBlockConsecutive;
      procedure NextBlockID(ABlockID : TBlockID);
      property Block : TBlockCounters read FBlock;
      property Trial : TTrialCounters read FTrial;
      property Blocks : Word read GetBlocks;
      property Trials : Word read GetTrials;
      property Tree : TIDSessionCounter read FTree;
    end;

implementation

uses session.pool, session.strutils;

{ TCustomUIDCounter }

function TCustomUIDCounter.ToString: string;
begin
  Result := inherited ToString;
end;

constructor TCustomUIDCounter.Create;
begin
  FEvents := TMutuallyExclusiveCounters.Create;
end;

destructor TCustomUIDCounter.Destroy;
begin
  FEvents.Free;
  inherited Destroy;
end;

procedure TCustomUIDCounter.Next;
begin
  inherited Next;
  FEvents.Invalidate;
end;

procedure TCustomUIDCounter.NextConsecutive;
begin
  inherited NextConsecutive;
  FEvents.Reset;
end;

{ TTrialCounters }

function TTrialCounters.ToString: string;
begin
  Result := KeyValue(CustomName, (UID+1).ToString) +
            KeyValue('ID', (ID+1).ToString) +
            inherited ToString + FEvents.ToString;
end;

procedure TTrialCounters.Invalidate;
begin
  inherited Invalidate;
  ID := 0;
end;

procedure TTrialCounters.NextID(AValue: TTrialID);
begin
  Next;
  ID := AValue;
end;

{ TBlockCounters }

function TBlockCounters.GetTrials: Word;
begin
  Result := FTrial.Count;
end;

function TBlockCounters.ToString: string;
begin
  Result := KeyValue(CustomName, (UID+1).ToString) +
            KeyValue('ID', (ID+1).ToString) +
            inherited ToString + FEvents.ToString;
end;

procedure TBlockCounters.Next;
begin
  inherited Next;
  FTrial.Invalidate;
end;

procedure TBlockCounters.NextID(AValue: TTrialID);
begin
  Next;
  ID := AValue;
end;

constructor TBlockCounters.Create;
begin
  inherited Create;
  FTrial := TTrialCounters.Create;
  FTrial.CustomName := 'Session.Block.Trial';
end;

destructor TBlockCounters.Destroy;
begin
  FTrial.Free;
  inherited Destroy;
end;

procedure TBlockCounters.Invalidate;
begin
  inherited Invalidate;
  ID := 0;
end;

procedure TBlockCounters.ResetConsecutive;
begin
  inherited ResetConsecutive;
  FTrial.Invalidate;
end;

{ TSessionCounters }

function TSessionCounters.GetTrials: Word;
begin
  Result := Block.Trial.UID;
end;

function TSessionCounters.GetBlocks: Word;
begin
  Result := FBlock.Count;
end;

procedure TSessionCounters.NextTrial;
begin
  Trial.Next;
end;

constructor TSessionCounters.Create;
begin
  inherited Create;
  CustomName := 'Session';
  FBlock := TBlockCounters.Create;
  FBlock.CustomName := 'Session.Block';
  FTrial := TTrialCounters.Create;
  FTrial.CustomName := 'Session.Trial';
  FTree.Initialize;
end;

destructor TSessionCounters.Destroy;
begin
  FTrial.Free;
  FBlock.Free;
  inherited Destroy;
end;

procedure TSessionCounters.Reset;
begin
  inherited Reset;
  FTrial.Reset;
  FBlock.Trial.Reset;
  FBlock.Reset;
end;

procedure TSessionCounters.NextID(AValue: TTrialID);
begin
  { Next Session ID }
end;

procedure TSessionCounters.NextTrialConsecutive;
begin
  FTrial.NextConsecutive;
  FBlock.Trial.NextConsecutive;
end;

procedure TSessionCounters.NextTrialID(ATrialID: TTrialID);
begin
  FTrial.Next; // session trials
  FBlock.Trial.NextID(ATrialID); // block trials
end;

procedure TSessionCounters.NextBlockID(ABlockID: TBlockID);
begin
  FBlock.NextID(ABlockID);
end;

end.

