{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.counters.consecutive;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

type

  { TConsecutivesCounter }

  TConsecutivesCounter = class(TObject)
    strict private
      FName : string;
      FCount : Word;
      FConsecutives : Word;
      FMaxConsecutives : Word;
    protected
      function GetCount : Word; virtual;
    public
      constructor Create;
      function ToIni: string; virtual;
      procedure LoadFromStream(AStream : TStream); virtual;
      procedure SaveToStream(AStream : TStream); virtual;
      procedure Next; virtual;
      procedure NextConsecutive; virtual;
      procedure Reset; virtual;
      procedure ResetConsecutive; virtual;
      procedure FlushMaxConsecutives;
      procedure Invalidate; virtual;
      property MaxConsecutives : Word read FMaxConsecutives;
      property Consecutives : Word read FConsecutives;
      property Count : Word read GetCount;
      property Name : string read FName write FName;
  end;

  { TUIDCounter }

  TUIDCounter = class(TConsecutivesCounter)
    strict private
      FUID : Word;
    public
      constructor Create;
      function ToIni : string; override;
      procedure LoadFromStream(AStream : TStream); override;
      procedure SaveToStream(AStream : TStream); override;
      procedure Next; override;
      procedure Invalidate; override;
      property UID : Word read FUID;
  end;

implementation

uses session.strutils;

function TConsecutivesCounter.ToIni: string;
begin
  Result :=
    KeyValue(FName+'.Count', FCount.ToString) +
    KeyValue(FName+'.Consecutives', FConsecutives.ToString) +
    KeyValue(FName+'.MaxConsecutives', FMaxConsecutives.ToString);
end;

function TConsecutivesCounter.GetCount: Word;
begin
  Result := FCount;
end;

procedure TConsecutivesCounter.LoadFromStream(AStream: TStream);
begin
  FCount := AStream.ReadWord;
  FConsecutives := AStream.ReadWord;
  FMaxConsecutives := AStream.ReadWord;
end;

procedure TConsecutivesCounter.SaveToStream(AStream: TStream);
begin
  AStream.WriteWord(FCount);
  AStream.WriteWord(FConsecutives);
  AStream.WriteWord(FMaxConsecutives);
end;

constructor TConsecutivesCounter.Create;
begin
  FName := TConsecutivesCounter.ClassName;
  Invalidate;
end;

procedure TConsecutivesCounter.Next;
begin
  Inc(FCount);
end;

procedure TConsecutivesCounter.Reset;
begin
  FCount := 0;
end;

procedure TConsecutivesCounter.NextConsecutive;
begin
  Inc(FConsecutives);
end;

procedure TConsecutivesCounter.ResetConsecutive;
begin
  FlushMaxConsecutives;
  FConsecutives := 0;
end;

procedure TConsecutivesCounter.FlushMaxConsecutives;
begin
  if FConsecutives > FMaxConsecutives then begin
    FMaxConsecutives := FConsecutives;
  end;
end;

procedure TConsecutivesCounter.Invalidate;
begin
  FCount := 0;
  FConsecutives := 0;
  FMaxConsecutives := 0;
end;

{ TUIDCounter }

function TUIDCounter.ToIni: string;
begin
  Result := KeyValue('UID', FUID.ToString) + inherited ToIni;
end;

constructor TUIDCounter.Create;
begin
  inherited Create;
  FUID := 0;
end;

procedure TUIDCounter.LoadFromStream(AStream: TStream);
begin
  inherited LoadFromStream(AStream);
  FUID := AStream.ReadWord;
end;

procedure TUIDCounter.SaveToStream(AStream: TStream);
begin
  inherited SaveToStream(AStream);
  AStream.WriteWord(FUID);
end;

procedure TUIDCounter.Next;
begin
  inherited Next;
  Inc(FUID);
end;

procedure TUIDCounter.Invalidate;
begin
  inherited Invalidate;
  FUID := 0;
end;

end.
