{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.choiceable.rectangule;

{$mode ObjFPC}{$H+}

interface

uses
  SysUtils, Generics.Collections
  , sdl.app.graphics.rectangule
  , sdl.app.choiceable.contract
  ;

type

  TChoices = specialize TList<TObject>;

  { TChoiceableRect }

  TChoiceableRect = class(TRectangule, IChoiceable)
  private
    FChoices : TChoices;
    function GetTargetChoice: TObject;
  protected

  public
    constructor Create; override;
    destructor Destroy; override;
    procedure AddOrderedChoice(AChoice: TObject);
    property TargetChoice : TObject read GetTargetChoice;
    property Choices : TChoices read FChoices;
  end;

implementation

{ TChoiceableRect }

function TChoiceableRect.GetTargetChoice: TObject;
begin
  if FChoices.Count > 0 then
    Result := FChoices[0] as TChoiceableRect
  else
    Result := nil;
end;

constructor TChoiceableRect.Create;
begin
  inherited Create;
  FChoices := TChoices.Create;
end;

destructor TChoiceableRect.Destroy;
begin
  FChoices.Free;
  inherited Destroy;
end;

procedure TChoiceableRect.AddOrderedChoice(AChoice: TObject);
begin
  FChoices.Add(AChoice);
end;

end.

