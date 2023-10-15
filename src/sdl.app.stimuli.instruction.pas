{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.instruction;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.events.abstract
  , sdl.app.stimuli
  , sdl.app.stimuli.contract
  , sdl.app.renderer.custom
  , sdl.app.graphics.text
  , sdl.app.stimulus.typeable;

type

  { TInstructionStimuli }

  TInstructionStimuli = class sealed (TStimuli, IStimuli)
    private
      FText : TText;
      FInstruction : TTypeableStimulus;
      procedure InstructionMouseDown(Sender: TObject;
        Shift: TCustomShiftState; X, Y: Integer);

    public
      constructor Create(AOwner : TComponent); override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses session.pool, sdl2_ttf;

{ TInstructionStimuli }

procedure TInstructionStimuli.InstructionMouseDown(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin
  DoExpectedResponse;
end;

constructor TInstructionStimuli.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);

end;

destructor TInstructionStimuli.Destroy;
begin
  inherited Destroy;
end;

function TInstructionStimuli.AsInterface: IStimuli;
begin
  Result := Self.AsInterface;
end;

procedure TInstructionStimuli.DoExpectedResponse;
begin
  if Assigned(OnFinalize) then
    OnFinalize(Self);
end;

procedure TInstructionStimuli.Load(AParameters: TStringList; AParent: TObject);
var
  Monitor : TSDL_Rect;
begin
  inherited Load(AParameters, AParent);
  Monitor := Pool.App.Monitor;
  FText := TText.Create(Self);
  FText.OnMouseDown := @InstructionMouseDown;
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 50;
  //FText.FontStyle := TTF_STYLE_UNDERLINE;
  FText.Wrapped := True;
  FText.WrappedWidth := (Monitor.w div 3) * 2;
  FText.LoadFromFile(AParameters.Values['Instruction']);
  FText.Parent := TCustomRenderer(AParent);
  FText.Centralize;
end;

procedure TInstructionStimuli.Start;
begin
  FText.Show;
end;

procedure TInstructionStimuli.Stop;
begin
  FText.Hide;
end;

end.

