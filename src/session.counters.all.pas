{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.all;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
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
    public
      constructor Create;
      function ToIni : string; override;
      procedure LoadFromStream(AStream : TStream); override;
      procedure SaveToStream(AStream : TStream); override;
      destructor Destroy; override;
      procedure NextID(AValue : TTrialID); virtual; abstract;
      property Events : TMutuallyExclusiveCounters read FEvents;
      property CustomName : string read FCustomName write FCustomName;
    end;

    { TTrialCounters }

    TTrialCounters = class(TCustomUIDCounter)
    public
      ID : TTrialID; static;
      function ToIni : string; override;
      procedure LoadFromStream(AStream : TStream); override;
      procedure SaveToStream(AStream : TStream); override;
      procedure Invalidate; override;
      procedure NextID(AValue : TTrialID); override;
    end;

    { TBlockCounters }

    TBlockCounters = class(TCustomUIDCounter)
    strict private
      FTrial : TTrialCounters;
    protected
      function GetTrials: Word; overload;
    public
      ID : TBlockID; static;
      constructor Create;
      destructor Destroy; override;
      function ToIni : string; override;
      procedure LoadFromStream(AStream : TStream); override;
      procedure SaveToStream(AStream : TStream); override;
      procedure Invalidate; override;
      procedure Next; override;
      procedure NextID(AValue : TTrialID); override;
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
    public
      ID : Word; static;
      constructor Create(AMockInitialization : Boolean = False);
      destructor Destroy; override;
      function ToIni : string; override;
      procedure LoadFromFile(AFilename : string);
      procedure LoadFromStream(AStream : TStream); override;
      procedure SaveToFile(AFilename : string);
      procedure SaveToStream(AStream : TStream); override;
      procedure NextID(AValue : TTrialID); override;
      procedure NextTrialConsecutive;
      procedure NextTrialID(ATrialID : TTrialID);
      procedure NextBlockID(ABlockID : TBlockID);
      procedure Reset; override;
      property Block : TBlockCounters read FBlock;
      property Trial : TTrialCounters read FTrial;
      property Blocks : Word read GetBlocks;
      property Trials : Word read GetTrials;
      property Tree : TIDSessionCounter read FTree;
    end;

implementation

uses session.pool, session.strutils;

{ TCustomUIDCounter }

function TCustomUIDCounter.ToIni: string;
begin
  Result :=
    KeyValue('CustomName', CustomName) +
    inherited ToIni + FEvents.ToIni;
end;

constructor TCustomUIDCounter.Create;
begin
  Name := TCustomUIDCounter.ClassName;
  FEvents := TMutuallyExclusiveCounters.Create;
end;

procedure TCustomUIDCounter.LoadFromStream(AStream: TStream);
begin
  inherited LoadFromStream(AStream);
  FCustomName := AStream.ReadAnsiString;
  FEvents.LoadFromStream(AStream);
end;

procedure TCustomUIDCounter.SaveToStream(AStream: TStream);
begin
  inherited SaveToStream(AStream);
  AStream.WriteAnsiString(FCustomName);
  FEvents.SaveToStream(AStream);
end;

destructor TCustomUIDCounter.Destroy;
begin
  FEvents.Free;
  inherited Destroy;
end;

{ TTrialCounters }

function TTrialCounters.ToIni: string;
begin
  Result := KeyValue(CustomName, (UID+1).ToString) +
            KeyValue('ID', ID.ToString) +
            inherited ToIni;
end;

procedure TTrialCounters.LoadFromStream(AStream: TStream);
begin
  inherited LoadFromStream(AStream);
  ID := AStream.ReadWord;
end;

procedure TTrialCounters.SaveToStream(AStream: TStream);
begin
  inherited SaveToStream(AStream);
  AStream.WriteWord(ID);
end;

procedure TTrialCounters.Invalidate;
begin
  inherited Invalidate;
  FEvents.Invalidate;
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

function TBlockCounters.ToIni: string;
begin
  Result := KeyValue(CustomName, (UID+1).ToString) +
            KeyValue('ID', ID.ToString) +
            inherited ToIni;
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

procedure TBlockCounters.LoadFromStream(AStream: TStream);
begin
  inherited LoadFromStream(AStream);
  ID := AStream.ReadWord;
  FTrial.LoadFromStream(AStream);
end;

procedure TBlockCounters.SaveToStream(AStream: TStream);
begin
  inherited SaveToStream(AStream);
  AStream.WriteWord(ID);
  FTrial.SaveToStream(AStream);
end;

procedure TBlockCounters.Invalidate;
begin
  inherited Invalidate;
  ID := 0;
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

function TSessionCounters.ToIni: string;
begin
  Result :=
    inherited ToIni +
    FBlock.ToIni +
    FBlock.Trial.ToIni +
    FTrial.ToIni;
end;

constructor TSessionCounters.Create(AMockInitialization: Boolean);
begin
  inherited Create;
  CustomName := 'Session';
  FBlock := TBlockCounters.Create;
  FBlock.CustomName := 'Session.Block';
  FTrial := TTrialCounters.Create;
  FTrial.CustomName := 'Session.Trial';
  if AMockInitialization then begin
    FTree.InitializeMock;
  end else begin
    FTree.Initialize;
  end;
end;

destructor TSessionCounters.Destroy;
begin
  FTrial.Free;
  FBlock.Free;
  inherited Destroy;
end;

procedure TSessionCounters.LoadFromFile(AFilename: string);
var
  LFileStream : TFileStream;
begin
  LFileStream := TFileStream.Create(AFilename, fmOpenRead);
  try
    LoadFromStream(LFileStream);
  finally
    LFileStream.Free;
  end;
end;

procedure TSessionCounters.LoadFromStream(AStream: TStream);
begin
  inherited LoadFromStream(AStream);
  ID := AStream.ReadWord;
  FBlock.LoadFromStream(AStream);
  FTrial.LoadFromStream(AStream);
  FTree.LoadFromStream(AStream);
end;

procedure TSessionCounters.SaveToFile(AFilename: string);
var
  LFileStream : TFileStream;
begin
  LFileStream := TFileStream.Create(AFilename, fmCreate);
  try
    SaveToStream(LFileStream);
  finally
    LFileStream.Free;
  end;
end;

procedure TSessionCounters.SaveToStream(AStream: TStream);
begin
  inherited SaveToStream(AStream);
  AStream.WriteWord(ID);
  FBlock.SaveToStream(AStream);
  FTrial.SaveToStream(AStream);
  FTree.SaveToStream(AStream);
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
  if AValue = ID then Exit;
  ID := AValue;
end;

procedure TSessionCounters.NextTrialConsecutive;
begin
  FTrial.NextConsecutive;
  FBlock.Trial.NextConsecutive;
end;

procedure TSessionCounters.NextTrialID(ATrialID: TTrialID);
begin
  FTrial.Next; // increment session trials
  FBlock.Trial.NextID(ATrialID); // increment block trials
end;

procedure TSessionCounters.NextBlockID(ABlockID: TBlockID);
begin
  FBlock.NextID(ABlockID);
end;

end.

