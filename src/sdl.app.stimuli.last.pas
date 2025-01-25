{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.last;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.graphics.text
  , sdl.app.stimuli
  , sdl.app.stimuli.contract;

type

  { TLastStimuli }

  TLastStimuli = class sealed (TStimuli)
    private
      FText : TText;
    public
      constructor Create; override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses
  sdl.app.controls.custom;

{ TLastStimuli }

constructor TLastStimuli.Create;
begin
  inherited Create;
  FText := TText.Create;
end;

destructor TLastStimuli.Destroy;
begin
  FText.Free;
  inherited Destroy;
end;

function TLastStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli;
end;

procedure TLastStimuli.DoExpectedResponse;
begin

end;

procedure TLastStimuli.Load(AParameters: TStringList; AParent: TObject);
var
  LText : string = 'Fim.';
begin
  inherited Load(AParameters, AParent);
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 150;
  FText.Load(LText);
  FText.CustomName := LText;
  FSelectables.Add(FText.AsISelectable);
  FText.Parent := TSDLControl(AParent);
  FText.Centralize;
end;

procedure TLastStimuli.Start;
begin
  FText.Show;
end;

procedure TLastStimuli.Stop;
begin
  FText.Hide;
end;

end.

