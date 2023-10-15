{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

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
      FCount : Word;
      FConsecutives : Word;
      FMaxConsecutives : Word;
    protected
      function ToString: string; virtual;
      function GetCount : Word; virtual;
    public
      constructor Create;
      procedure Next; virtual;
      procedure NextConsecutive; virtual;
      procedure Reset; virtual;
      procedure ResetConsecutive; virtual;
      procedure FlushMaxConsecutives;
      procedure Invalidate; virtual;
      property MaxConsecutives : Word read FMaxConsecutives;
      property Consecutives : Word read FConsecutives;
      property Count : Word read GetCount;
  end;

  { TUIDCounter }

  TUIDCounter = class(TConsecutivesCounter)
    strict private
      FUID : Word;
    protected
      function ToString : string; override;
    public
      constructor Create;
      procedure Next; override;
      procedure Invalidate; virtual;
      property UID : Word read FUID;
  end;

implementation

uses session.strutils;

function TConsecutivesCounter.ToString: string;
begin
  Result :=
    KeyValue('Count', FCount.ToString) +
    KeyValue('Consecutives', FConsecutives.ToString) +
    KeyValue('MaxConsecutives', FMaxConsecutives.ToString);
end;

function TConsecutivesCounter.GetCount: Word;
begin
  Result := FCount;
end;

constructor TConsecutivesCounter.Create;
begin
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

function TUIDCounter.ToString: string;
begin
  Result := KeyValue('UID', FUID.ToString) + inherited ToString;
end;

constructor TUIDCounter.Create;
begin
  inherited Create;
  FUID := 0;
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

