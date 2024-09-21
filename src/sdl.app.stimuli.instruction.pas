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
  , sdl.app.navigable.contract
  , sdl.app.navigator.contract
  , sdl.app.selectable.list
  , sdl.app.controls.custom
  , sdl.app.graphics.text
  , sdl.app.stimulus.typeable;

type

  { TInstructionStimuli }

  TInstructionStimuli = class sealed (TStimuli, INavigable)
    private
      FNavigator : ITableNavigator;
      FText : TText;
      //FInstruction : TTypeableStimulus;
      procedure InstructionMouseUp(Sender: TObject;
        Shift: TCustomShiftState; X, Y: Integer);
      procedure SetNavigator(ANavigator: ITableNavigator);
      procedure UpdateNavigator;
    public
      constructor Create; override;
      destructor Destroy; override;
      function AsINavigable: INavigable; override;
      function IsStarter : Boolean; override;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses session.pool;

{ TInstructionStimuli }

procedure TInstructionStimuli.InstructionMouseUp(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin
  DoExpectedResponse;
end;

procedure TInstructionStimuli.SetNavigator(ANavigator: ITableNavigator);
begin
  FNavigator := ANavigator;
end;

procedure TInstructionStimuli.UpdateNavigator;
var
  LControls : TSelectables;
begin
  LControls := TSelectables.Create;
  LControls.Add(FText.AsISelectable);
  try
    FNavigator.UpdateNavigationControls(LControls);
  finally
    LControls.Free;
  end;
end;

constructor TInstructionStimuli.Create;
begin
  inherited Create;
  FText := TText.Create;
end;

destructor TInstructionStimuli.Destroy;
begin
  FText.Free;
  inherited Destroy;
end;

function TInstructionStimuli.AsINavigable: INavigable;
begin
  Result := Self as INavigable;
end;

function TInstructionStimuli.IsStarter: Boolean;
begin
  Result := True;
end;

procedure TInstructionStimuli.DoExpectedResponse;
begin
  Finalize;
end;

procedure TInstructionStimuli.Load(AParameters: TStringList; AParent: TObject);
var
  Monitor : TSDL_Rect;
  LInstruction : string;
begin
  inherited Load(AParameters, AParent);
  LInstruction := AParameters.Values['Instruction'];

  Monitor := Pool.App.Monitor;
  FText.OnMouseUp := @InstructionMouseUp;
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 50;
  //FText.FontStyle := TTF_STYLE_UNDERLINE;
  FText.Wrapped := True;
  FText.WrappedWidth := (Monitor.w div 3) * 2;
  FText.LoadFromFile(LInstruction);
  FText.Parent := TSDLControl(AParent);
  FText.Centralize;
  FText.CustomName := LInstruction;
  FSelectables.Add(FText.AsISelectable);
end;

procedure TInstructionStimuli.Start;
begin
  FText.Show;
  UpdateNavigator;
end;

procedure TInstructionStimuli.Stop;
begin
  FText.Hide;
end;

end.

