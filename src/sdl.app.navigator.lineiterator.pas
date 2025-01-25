{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.navigator.lineiterator;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils,
  Generics.Aggregator,
  sdl.app.selectable.contract,
  sdl.app.selectable.list;

type

  { TPossibleSelections }

  TPossibleSelections = class (specialize TAggregator<ISelectable>)
  private
    FFirst : Boolean;
    FBaseControl : ISelectable;
  public
    function Next : ISelectable;
    function Previous : ISelectable;
    procedure Update(ASelectables : TSelectables);
    procedure SetBaseControl(ABaseControl : ISelectable);
  end;


implementation


{ TPossibleSelections }

function TPossibleSelections.Next: ISelectable;
begin
  if List.Count = 0 then begin
    Result := FBaseControl;
    if Result = nil then begin
      raise EArgumentNilException.Create('Base control = nil');
    end;
    Exit;
  end;

  if FFirst then begin
    Result := Iterator.GetCurrent;
    FFirst := False;
  end else begin
    with Iterator do begin
      if IsLast then begin
        GoFirst;
      end else begin
        GoNext;
      end;
      Result := GetCurrent;
    end;
  end;
end;

function TPossibleSelections.Previous: ISelectable;
begin
  if List.Count = 0 then begin
    Result := FBaseControl;
    if Result = nil then begin
      raise EArgumentNilException.Create('Base control = nil');
    end;
    Exit;
  end;

  if FFirst then begin
    Result := Iterator.GetCurrent;
    FFirst := False;
  end else begin
    with Iterator do begin
      if IsFirst then begin
        GoLast;
      end else begin
        GoPrevious;
      end;
      Result := GetCurrent;
    end;
  end;
end;

procedure TPossibleSelections.Update(ASelectables: TSelectables);
var
  LISelectable : ISelectable;
begin
  FFirst := True;
  List.Clear;
  if ASelectables = nil then Exit;
  for LISelectable in ASelectables do begin
    List.Add(LISelectable);
  end;
  Iterator.GoFirst;
end;

procedure TPossibleSelections.SetBaseControl(ABaseControl: ISelectable);
begin
  FBaseControl := ABaseControl;
end;

end.

