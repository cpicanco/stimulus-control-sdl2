{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.hierarquical;

{$mode ObjFPC}{$H+}
{$modeswitch AdvancedRecords}

interface

uses
  Classes, SysUtils;

type

  { TIDCounter }

  TIDCounter = record
    Count : SmallInt;
    procedure Increment;
    procedure SaveToStream(AStream : TStream);
    procedure LoadFromStream(AStream : TStream);
  end;

  TIDTrialCounts = array of TIDCounter;

  { TIDBlockCounter }

  TIDBlockCounter = record
    Count : SmallInt;
    Trial : TIDTrialCounts;
    procedure Increment;
    procedure SaveToStream(AStream : TStream);
    procedure LoadFromStream(AStream : TStream);
  end;

  TIDBlockCounts = array of TIDBlockCounter;

  { TIDSessionCounter }

  TIDSessionCounter = record
    Block : TIDBlockCounts;
    procedure Initialize;
    procedure InitializeMock;
    procedure SaveToStream(AStream : TStream);
    procedure LoadFromStream(AStream : TStream);
  end;

implementation

uses session.configurationfile;

{ TIDCounter }

procedure TIDCounter.Increment;
begin
  Inc(Count);
end;

procedure TIDCounter.SaveToStream(AStream: TStream);
begin
  AStream.Write(Count, SizeOf(SmallInt));
end;

procedure TIDCounter.LoadFromStream(AStream: TStream);
begin
  AStream.Read(Count, SizeOf(SmallInt));
end;

{ TIDBlockCounter }

procedure TIDBlockCounter.Increment;
begin
  Inc(Count);
end;

procedure TIDBlockCounter.SaveToStream(AStream: TStream);
var
  LTrials , i: Integer;
begin
  AStream.Write(Count, SizeOf(SmallInt));
  LTrials := Length(Trial);
  AStream.Write(LTrials, SizeOf(Integer));
  if LTrials > 0 then begin
    for i := Low(Trial) to High(Trial) do begin
      Trial[i].SaveToStream(AStream);
    end;
  end;
end;

procedure TIDBlockCounter.LoadFromStream(AStream: TStream);
var
  LTrials : Integer = 0;
  i : Integer;
begin
  AStream.Read(Count, SizeOf(SmallInt));
  AStream.Read(LTrials, SizeOf(Integer));
  LTrials := Length(Trial);
  if LTrials > 0 then begin
    for i := Low(Trial) to High(Trial) do begin
      Trial[i].LoadFromStream(AStream);
    end;
  end;
end;

{ TIDSessionCounter }

procedure TIDSessionCounter.Initialize;
var
  i, j: Integer;
begin
  Block := Default(TIDBlockCounts);
  SetLength(Block, ConfigurationFile.TotalBlocks);
  for i := Low(Block) to High(Block) do begin
    Block[i].Count := 0;
    SetLength(Block[i].Trial, ConfigurationFile.Block[i].TotalTrials);
    for j := Low(Block[i].Trial) to High(Block[i].Trial) do begin
      Block[i].Trial[j].Count := 0;
    end;
  end;
end;

procedure TIDSessionCounter.InitializeMock;
var
  i, j: Integer;
begin
  Block := Default(TIDBlockCounts);
  SetLength(Block, 10);
  for i := Low(Block) to High(Block) do begin
    Block[i].Count := Random(Length(Block));
    SetLength(Block[i].Trial, 6);
    for j := Low(Block[i].Trial) to High(Block[i].Trial) do begin
      Block[i].Trial[j].Count := Random(Length(Block[i].Trial));
    end;
  end;
end;

procedure TIDSessionCounter.SaveToStream(AStream: TStream);
var
  LBlocks , i: Integer;
begin
  LBlocks := Length(Block);
  AStream.Write(LBlocks, SizeOf(Integer));
  if LBlocks > 0 then begin
    for i := Low(Block) to High(Block) do begin
      Block[i].SaveToStream(AStream);
    end;
  end;
end;

procedure TIDSessionCounter.LoadFromStream(AStream: TStream);
var
  LBlocks : Integer = 0;
  i: Integer;
begin
  AStream.Read(LBlocks, SizeOf(Integer));
  LBlocks := Length(Block);
  if LBlocks > 0 then begin
    for i := Low(Block) to High(Block) do begin
      Block[i].LoadFromStream(AStream);
    end;
  end;
end;

end.

